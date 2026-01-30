"""
Neural Field PDE Solver for KATIE-style NT Dynamics.

Biological Basis:
- Neural mass models (Jansen-Rit, Wilson-Cowan)
- Spatiotemporal dynamics across brain regions
- 6 neurotransmitter fields: DA, 5-HT, ACh, NE, GABA, Glu

Core Equation:
    ∂U(x,t)/∂t = -αU + D∇²U + S(x,t) + C(U₁...Uₙ)

where:
    U = neurotransmitter concentration field
    α = decay rate (NT-specific)
    D = diffusion coefficient
    S = external stimulus
    C = coupling function (from coupling.py)

Implementation designed by CompBio agent with biological plausibility focus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import laplace

if TYPE_CHECKING:
    import torch

    from ww.nca.adenosine import AdenosineDynamics
    from ww.nca.astrocyte import AstrocyteLayer
    from ww.nca.connectome import Connectome, ConnectomeIntegrator
    from ww.nca.delays import TransmissionDelaySystem
    from ww.nca.oscillators import FrequencyBandGenerator
    from ww.nca.striatal_coupling import DAACHCoupling

logger = logging.getLogger(__name__)


class NumericalInstabilityError(RuntimeError):
    """Raised when numerical instability (NaN/Inf) is detected in neural field."""
    pass


class NeurotransmitterType(Enum):
    """Six-neurotransmitter system following KATIE specification."""
    DOPAMINE = auto()
    SEROTONIN = auto()
    ACETYLCHOLINE = auto()
    NOREPINEPHRINE = auto()
    GABA = auto()
    GLUTAMATE = auto()


@dataclass
class NeurotransmitterState:
    """
    6-NT state vector with biological plausibility.

    All concentrations normalized to [0, 1] representing
    proportion of maximum physiological range.
    """
    dopamine: float = 0.5
    serotonin: float = 0.5
    acetylcholine: float = 0.5
    norepinephrine: float = 0.5
    gaba: float = 0.5
    glutamate: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [DA, 5HT, ACh, NE, GABA, Glu]."""
        return np.array([
            self.dopamine, self.serotonin, self.acetylcholine,
            self.norepinephrine, self.gaba, self.glutamate
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> NeurotransmitterState:
        """Create from numpy array."""
        return cls(
            dopamine=float(arr[0]),
            serotonin=float(arr[1]),
            acetylcholine=float(arr[2]),
            norepinephrine=float(arr[3]),
            gaba=float(arr[4]),
            glutamate=float(arr[5])
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        import torch
        return torch.tensor(self.to_array(), dtype=torch.float32)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> NeurotransmitterState:
        """Create from PyTorch tensor."""
        return cls.from_array(t.detach().cpu().numpy())

    def __post_init__(self):
        """Clamp all values to biological range [0, 1]."""
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        self.acetylcholine = np.clip(self.acetylcholine, 0.0, 1.0)
        self.norepinephrine = np.clip(self.norepinephrine, 0.0, 1.0)
        self.gaba = np.clip(self.gaba, 0.0, 1.0)
        self.glutamate = np.clip(self.glutamate, 0.0, 1.0)


@dataclass
class NeuralFieldConfig:
    """
    Configuration for neural field PDE solver.

    Biological parameters based on CompBio analysis of
    NT diffusion and decay rates.
    """
    # Decay rates (1/s) - how fast NT is cleared
    # Based on neuroscience literature:
    # - Glu/GABA: Fast (synaptic), ~10-100ms clearance
    # - ACh: Fast (cholinergic), ~50ms
    # - DA: Moderate (dopaminergic), ~100ms
    # - NE: Moderate (noradrenergic), ~200ms
    # - 5-HT: Slow (serotonergic), ~seconds
    alpha_da: float = 10.0    # DA: ~100ms timescale (1/0.1)
    alpha_5ht: float = 2.0    # 5-HT: ~500ms timescale (1/0.5)
    alpha_ach: float = 20.0   # ACh: ~50ms timescale (1/0.05)
    alpha_ne: float = 5.0     # NE: ~200ms timescale (1/0.2)
    alpha_gaba: float = 100.0 # GABA: ~10ms timescale (1/0.01)
    alpha_glu: float = 200.0  # Glu: ~5ms timescale (1/0.005), prevent excitotoxicity

    # Diffusion coefficients (mm²/s) - spatial spread
    # Volume transmission estimates from literature
    diffusion_da: float = 0.1
    diffusion_5ht: float = 0.2   # 5-HT has wider influence
    diffusion_ach: float = 0.05  # ACh is more localized
    diffusion_ne: float = 0.15   # NE is diffuse neuromodulator
    diffusion_gaba: float = 0.03 # GABA is local (synaptic)
    diffusion_glu: float = 0.02  # Glu is highly local (prevent excitotoxicity)

    # Integration parameters
    dt: float = 0.001        # Time step (s) - 1ms for stability
    dt_max: float = 0.01     # Maximum adaptive time step
    dt_min: float = 0.0001   # Minimum adaptive time step
    spatial_dims: int = 1    # 1D, 2D, or 3D field
    grid_size: int = 32      # Points per dimension
    dx: float = 1.0          # Spatial step (mm)
    splitting_method: str = "strang"  # "euler" or "strang" splitting

    # Numerical stability
    max_concentration: float = 1.0
    min_concentration: float = 0.0
    adaptive_timestepping: bool = True
    stability_threshold: float = 0.5  # CFL condition: D*dt/dx² < 0.5

    # Boundary conditions
    boundary_type: str = "no-flux"  # "no-flux", "periodic", or "dirichlet"

    # P1-3: GABA lateral inhibition parameters
    gaba_lateral_inhibition: bool = True  # Enable GABA-mediated lateral inhibition
    gaba_inhibition_strength: float = 0.3  # Strength of GABA → Glu inhibition
    gaba_surround_sigma: float = 2.0  # Spatial extent of surround suppression (grid units)

    def __post_init__(self):
        """Validate configuration and adjust for stability."""
        # Check CFL condition for stability
        max_alpha = max([
            self.alpha_da, self.alpha_5ht, self.alpha_ach,
            self.alpha_ne, self.alpha_gaba, self.alpha_glu
        ])
        max_diffusion = max([
            self.diffusion_da, self.diffusion_5ht, self.diffusion_ach,
            self.diffusion_ne, self.diffusion_gaba, self.diffusion_glu
        ])

        # CFL condition: dt < dx²/(2*D*spatial_dims)
        cfl_dt = self.dx**2 / (2 * max_diffusion * self.spatial_dims + 1e-10)

        if self.dt > cfl_dt:
            logger.warning(
                f"dt={self.dt} exceeds CFL stability limit {cfl_dt:.6f}. "
                f"Consider reducing dt or enabling adaptive timestepping."
            )

        # Check that decay doesn't make system too stiff (only if decay > 0)
        if max_alpha > 0:
            max_decay_dt = 1.0 / max_alpha
            if self.dt > max_decay_dt * 10:
                logger.warning(
                    f"dt={self.dt} may be too large for decay rates. "
                    f"Fastest decay timescale: {max_decay_dt:.6f}s"
                )


class NeuralFieldSolver:
    """
    PDE solver for 6-NT neural field dynamics.

    Implements KATIE's core equation with coupling from LearnableCoupling.
    Uses semi-implicit Euler for stability with fast decay rates.

    Numerical Method:
    - Semi-implicit Euler: decay treated implicitly, rest explicitly
    - Laplacian computed via finite differences or scipy.ndimage
    - Adaptive timestepping for stiff systems
    - No-flux boundary conditions by default
    """

    def __init__(
        self,
        config: NeuralFieldConfig | None = None,
        coupling: LearnableCoupling | None = None,
        attractor_manager: StateTransitionManager | None = None,
        da_ach_coupling: DAACHCoupling | None = None,
        astrocyte_layer: AstrocyteLayer | None = None,
        oscillator: FrequencyBandGenerator | None = None,
        adenosine: AdenosineDynamics | None = None,
        delay_system: TransmissionDelaySystem | None = None,
        connectome: Connectome | None = None
    ):
        """
        Initialize neural field solver.

        Args:
            config: PDE solver configuration
            coupling: Learnable coupling matrix (from coupling.py)
            attractor_manager: State transition manager (from attractors.py)
            da_ach_coupling: DA-ACh striatal coupling for phase-lagged dynamics
            astrocyte_layer: Astrocyte glial layer for Glu/GABA reuptake
            oscillator: Frequency band generator for theta/gamma oscillations
            adenosine: Adenosine sleep-wake dynamics for fatigue/consolidation
            delay_system: Transmission delay system for axonal/synaptic delays
            connectome: Anatomical brain connectome for region-specific modulation
        """
        self.config = config or NeuralFieldConfig()
        self.coupling = coupling
        self.attractor_manager = attractor_manager
        self.da_ach_coupling = da_ach_coupling
        self.astrocyte_layer = astrocyte_layer
        self.oscillator = oscillator
        self.adenosine = adenosine
        self.delay_system = delay_system
        self.connectome = connectome
        self._connectome_integrator: ConnectomeIntegrator | None = None

        # Configure systems from connectome if provided
        if connectome is not None:
            self._configure_from_connectome()

        # Initialize fields
        self._init_fields()

        # Track simulation state
        self._time = 0.0
        self._step_count = 0
        self._rejected_steps = 0

        # Adaptive timestepping
        self._current_dt = self.config.dt

        # Get parameter arrays
        self._alphas = self._get_decay_array()
        self._diffusions = self._get_diffusion_array()

        logger.info(
            f"NeuralFieldSolver initialized: "
            f"dims={self.config.spatial_dims}, "
            f"grid={self.config.grid_size}, "
            f"dt={self.config.dt}, "
            f"da_ach_coupling={da_ach_coupling is not None}, "
            f"astrocyte={astrocyte_layer is not None}, "
            f"oscillator={oscillator is not None}, "
            f"adenosine={adenosine is not None}, "
            f"delay_system={delay_system is not None}, "
            f"connectome={connectome is not None}"
        )

    def _init_fields(self) -> None:
        """Initialize NT concentration fields at baseline."""
        shape = (6,) + (self.config.grid_size,) * self.config.spatial_dims
        # Initialize at resting state (0.5 = baseline for all NTs)
        self.fields = np.full(shape, 0.5, dtype=np.float32)

    def _configure_from_connectome(self) -> None:
        """Configure delay and coupling systems from connectome."""
        from ww.nca.connectome import ConnectomeIntegrator

        self._connectome_integrator = ConnectomeIntegrator(self.connectome)

        # Configure delay system with connectome distances
        if self.delay_system is not None:
            try:
                self._connectome_integrator.configure_delay_system(self.delay_system)
                logger.debug("Delay system configured from connectome")
            except Exception as e:
                logger.warning(f"Could not configure delay system from connectome: {e}")

        # Configure coupling with connectome constraints
        if self.coupling is not None:
            try:
                self._connectome_integrator.configure_coupling(self.coupling)
                logger.debug("Coupling configured from connectome")
            except Exception as e:
                logger.warning(f"Could not configure coupling from connectome: {e}")

    def get_connectome_stats(self) -> dict:
        """Get connectome statistics if available."""
        if self.connectome is None:
            return {}
        return self.connectome.get_stats()

    def get_region_names(self) -> list:
        """Get list of brain region names from connectome."""
        if self.connectome is None:
            return []
        return self.connectome.get_region_names()

    def get_nt_sources(self, nt_name: str) -> list:
        """Get regions that produce a specific neurotransmitter."""
        if self.connectome is None:
            return []
        from ww.nca.connectome import NTSystem
        nt_map = {
            "dopamine": NTSystem.DOPAMINE,
            "serotonin": NTSystem.SEROTONIN,
            "acetylcholine": NTSystem.ACETYLCHOLINE,
            "norepinephrine": NTSystem.NOREPINEPHRINE,
            "glutamate": NTSystem.GLUTAMATE,
            "gaba": NTSystem.GABA,
        }
        nt = nt_map.get(nt_name.lower())
        if nt is None:
            return []
        return self.connectome.get_nt_sources(nt)

    def _get_decay_array(self) -> np.ndarray:
        """Get decay rates as shaped array for broadcasting."""
        alphas = np.array([
            self.config.alpha_da,
            self.config.alpha_5ht,
            self.config.alpha_ach,
            self.config.alpha_ne,
            self.config.alpha_gaba,
            self.config.alpha_glu,
        ], dtype=np.float32)

        # Reshape for broadcasting: [6, 1, 1, ...] depending on spatial_dims
        return alphas.reshape(6, *([1] * self.config.spatial_dims))

    def _get_diffusion_array(self) -> np.ndarray:
        """Get diffusion coefficients as shaped array."""
        diffusions = np.array([
            self.config.diffusion_da,
            self.config.diffusion_5ht,
            self.config.diffusion_ach,
            self.config.diffusion_ne,
            self.config.diffusion_gaba,
            self.config.diffusion_glu,
        ], dtype=np.float32)

        return diffusions.reshape(6, *([1] * self.config.spatial_dims))

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian (∇²) for diffusion operator.

        Uses scipy.ndimage.laplace for robust computation with
        proper boundary handling.

        Args:
            field: NT field to compute Laplacian on [grid_size, ...]

        Returns:
            Laplacian of field with same shape
        """
        # scipy.ndimage.laplace handles boundary conditions
        # mode='nearest' implements no-flux (Neumann) boundaries
        # mode='wrap' implements periodic boundaries
        mode = 'wrap' if self.config.boundary_type == 'periodic' else 'nearest'

        # Laplacian in physical units: ∇²U / dx²
        lap = laplace(field, mode=mode) / (self.config.dx ** 2)

        return lap

    def _compute_diffusion(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute diffusion term D∇²U for all NT fields.

        Args:
            fields: All NT fields [6, grid_size, ...]

        Returns:
            Diffusion contribution [6, grid_size, ...]
        """
        diffusion_term = np.zeros_like(fields)

        for i in range(6):
            # Compute Laplacian for this NT
            lap = self._compute_laplacian(fields[i])

            # Scale by diffusion coefficient
            diffusion_term[i] = self._diffusions[i] * lap

        return diffusion_term

    def _compute_coupling_field(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute coupling term C(U) across spatial field.

        Args:
            fields: Current NT fields [6, grid_size, ...]

        Returns:
            Coupling contribution [6, grid_size, ...]
        """
        if self.coupling is None:
            return np.zeros_like(fields)

        # Coupling is computed point-wise across spatial field
        coupling_field = np.zeros_like(fields)

        # Flatten spatial dimensions for vectorized computation
        spatial_shape = fields.shape[1:]
        flat_fields = fields.reshape(6, -1)  # [6, N_points]

        # Compute coupling for each spatial point
        for i in range(flat_fields.shape[1]):
            point_state = flat_fields[:, i]
            coupling_contribution = self.coupling.compute_coupling(point_state)
            coupling_field.reshape(6, -1)[:, i] = coupling_contribution

        return coupling_field

    def _compute_attractor_force(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute attractor basin force pulling toward current state.

        Args:
            fields: Current NT fields [6, grid_size, ...]

        Returns:
            Attractor force [6, grid_size, ...]
        """
        if self.attractor_manager is None:
            return np.zeros_like(fields)

        # Attractor force is spatially uniform (global brain state)
        mean_state = np.mean(fields, axis=tuple(range(1, fields.ndim)))
        force_vector = self.attractor_manager.get_attractor_force(mean_state)

        # Broadcast to spatial field
        force_field = force_vector.reshape(6, *([1] * self.config.spatial_dims))
        force_field = np.broadcast_to(force_field, fields.shape).copy()

        return force_field

    def _compute_da_ach_coupling(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute DA-ACh striatal coupling with phase lag.

        This implements bidirectional coupling between dopamine and acetylcholine
        with ~100ms phase lag, creating traveling wave dynamics essential for:
        - Habit formation
        - Reward-attention interaction
        - Learning gating

        Args:
            fields: Current NT fields [6, grid_size, ...]
                   Index 0 = DA, Index 2 = ACh

        Returns:
            DA-ACh coupling contribution [6, grid_size, ...]
        """
        if self.da_ach_coupling is None:
            return np.zeros_like(fields)

        # Get DA (index 0) and ACh (index 2) fields
        da_field = fields[0]  # [grid_size, ...]
        ach_field = fields[2]  # [grid_size, ...]

        # Compute coupling effects across spatial field
        da_effect, ach_effect = self.da_ach_coupling.compute_coupling_field(
            da_field, ach_field
        )

        # Create output with effects only on DA and ACh
        coupling_field = np.zeros_like(fields)
        coupling_field[0] = da_effect   # Effect on DA
        coupling_field[2] = ach_effect  # Effect on ACh

        return coupling_field

    def _compute_astrocyte_reuptake(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute astrocyte-mediated glutamate and GABA reuptake.

        Astrocytes:
        - Clear ~90% of synaptic glutamate via EAAT-2 (prevents excitotoxicity)
        - Clear GABA via GAT-3 (modulates inhibitory tone)
        - Release gliotransmitters (Glu, D-serine, ATP)

        Args:
            fields: Current NT fields [6, grid_size, ...]
                   Index 4 = GABA, Index 5 = Glutamate

        Returns:
            Reuptake contribution (negative for cleared NTs) [6, grid_size, ...]
        """
        if self.astrocyte_layer is None:
            return np.zeros_like(fields)

        # Get Glutamate (index 5) and GABA (index 4) fields
        glu_field = fields[5]  # [grid_size, ...]
        gaba_field = fields[4]  # [grid_size, ...]

        # Compute mean activity for astrocyte modulation
        activity_field = np.mean(fields, axis=0)

        # Compute reuptake via astrocyte
        glu_cleared, gaba_cleared = self.astrocyte_layer.compute_reuptake_field(
            glu_field, gaba_field, activity_field
        )

        # Create output with reuptake as negative contribution
        # (reduces Glu and GABA concentrations)
        reuptake_field = np.zeros_like(fields)
        reuptake_field[5] = -glu_cleared   # Remove glutamate
        reuptake_field[4] = -gaba_cleared  # Remove GABA

        # Add gliotransmission (astrocyte releases small amounts of Glu)
        gliotx = self.astrocyte_layer.compute_gliotransmission()
        if gliotx["glutamate"] > 0:
            # Broadcast gliotransmitter release across field
            reuptake_field[5] += gliotx["glutamate"] * 0.1  # Small release

        return reuptake_field

    def _compute_gaba_lateral_inhibition(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute GABA-mediated spatial lateral inhibition.

        P1-3: Implements biologically-plausible center-surround inhibition
        where local GABA activity inhibits surrounding glutamatergic activity.

        Biological basis: GABA interneurons (parvalbumin+, somatostatin+)
        provide local inhibitory surround suppression, sharpening neural
        representations through winner-take-all competition.

        Args:
            fields: Current NT fields [6, grid_size, ...]
                   Index 4 = GABA, Index 5 = Glutamate

        Returns:
            Lateral inhibition contribution [6, grid_size, ...]
            (negative effect on Glu where GABA is high)
        """
        if not self.config.gaba_lateral_inhibition:
            return np.zeros_like(fields)

        gaba_field = fields[4]  # [grid_size, ...]
        glu_field = fields[5]   # [grid_size, ...]

        # Create surround suppression kernel (Mexican hat / difference of Gaussians)
        # For 1D: simple neighbor-based inhibition
        # For 2D/3D: proper Gaussian kernel
        if self.config.spatial_dims == 1:
            # 1D lateral inhibition: high GABA inhibits nearby Glu
            sigma = self.config.gaba_surround_sigma
            kernel_size = int(4 * sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create Gaussian kernel for surround suppression
            x = np.arange(kernel_size) - kernel_size // 2
            surround_kernel = np.exp(-x**2 / (2 * sigma**2))
            surround_kernel = surround_kernel / surround_kernel.sum()

            # Convolve GABA field with kernel to get local inhibition
            from scipy.ndimage import convolve1d
            gaba_spread = convolve1d(gaba_field, surround_kernel, mode='nearest')

        else:
            # 2D/3D: use scipy.ndimage for proper convolution
            from scipy.ndimage import gaussian_filter
            sigma = self.config.gaba_surround_sigma
            gaba_spread = gaussian_filter(gaba_field, sigma=sigma, mode='nearest')

        # Compute inhibitory effect on glutamate
        # Higher local GABA → stronger inhibition of Glu
        inhibition = self.config.gaba_inhibition_strength * gaba_spread * glu_field

        # Create output field (only affects Glu)
        lateral_field = np.zeros_like(fields)
        lateral_field[5] = -inhibition  # Reduce glutamate where GABA is active

        return lateral_field

    def _compute_oscillation_drive(self, fields: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute oscillatory drive from frequency band generator.

        Oscillations modulate NT dynamics:
        - Theta modulates ACh (index 2)
        - Gamma modulates Glu (index 5)
        - Beta modulates DA (index 0)

        Args:
            fields: Current NT fields [6, grid_size, ...]
            dt: Timestep in seconds

        Returns:
            Oscillation contribution [6, grid_size, ...]
        """
        if self.oscillator is None:
            return np.zeros_like(fields)

        # Get current NT levels (spatial means for oscillator state)
        ach_field = fields[2]   # ACh
        da_field = fields[0]    # DA
        glu_field = fields[5]   # Glu
        gaba_field = fields[4]  # GABA

        # Convert dt from seconds to ms for oscillator
        dt_ms = dt * 1000.0

        # Compute oscillations across spatial field
        osc = self.oscillator.compute_oscillation_field(
            ach_field, da_field, glu_field, gaba_field, dt_ms
        )

        # Create output with oscillation drives
        osc_field = np.zeros_like(fields)

        # Theta drives ACh fluctuations
        osc_field[2] = osc["theta"] * 0.1  # Scale factor

        # Gamma drives Glu fluctuations
        osc_field[5] = osc["gamma"] * 0.05

        # Beta drives DA fluctuations
        osc_field[0] = osc["beta"] * 0.08

        return osc_field

    def get_cognitive_phase(self) -> str:
        """
        Get current cognitive phase from oscillator.

        Returns:
            'encoding' or 'retrieval' based on theta phase
        """
        if self.oscillator is None:
            return "unknown"
        return self.oscillator.state.cognitive_phase.value

    def get_encoding_signal(self) -> float:
        """
        Get encoding strength (0-1) for plasticity modulation.

        High during theta encoding phase, low during retrieval.
        """
        if self.oscillator is None:
            return 0.5
        return self.oscillator.get_encoding_signal()

    def _compute_adenosine_modulation(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute adenosine-based modulation of NT dynamics.

        Adenosine (sleep pressure) suppresses wake-promoting NTs
        and potentiates GABA. This models fatigue and drives
        consolidation need.

        Args:
            fields: Current NT fields [6, grid_size, ...]

        Returns:
            Adenosine modulation [6, grid_size, ...]
        """
        if self.adenosine is None:
            return np.zeros_like(fields)

        # Get modulation factors from adenosine dynamics
        modulation = self.adenosine.get_nt_modulation()

        # Create modulation field
        mod_field = np.zeros_like(fields)

        # Apply modulation as multiplicative factor relative to baseline
        # These are relative changes, not absolute - we modulate toward baseline*factor
        baseline = 0.5

        # DA suppression (index 0): high adenosine reduces DA
        da_target = baseline * modulation["da"]
        mod_field[0] = (da_target - fields[0]) * 0.1

        # NE suppression (index 3): high adenosine reduces NE
        ne_target = baseline * modulation["ne"]
        mod_field[3] = (ne_target - fields[3]) * 0.1

        # ACh suppression (index 2): high adenosine reduces ACh
        ach_target = baseline * modulation["ach"]
        mod_field[2] = (ach_target - fields[2]) * 0.05

        # GABA potentiation (index 4): high adenosine increases GABA
        gaba_target = baseline * modulation["gaba"]
        mod_field[4] = (gaba_target - fields[4]) * 0.1

        # Serotonin mild suppression (index 1)
        sero_target = baseline * modulation["5ht"]
        mod_field[1] = (sero_target - fields[1]) * 0.02

        return mod_field

    def get_sleep_pressure(self) -> float:
        """
        Get current sleep pressure from adenosine system.

        Returns:
            Sleep pressure 0-1 (0=alert, 1=exhausted)
        """
        if self.adenosine is None:
            return 0.0
        return self.adenosine.state.sleep_pressure

    def should_consolidate(self) -> bool:
        """
        Check if system should trigger consolidation.

        Returns:
            True if adenosine indicates sleep/consolidation needed
        """
        if self.adenosine is None:
            return False
        return self.adenosine.should_sleep()

    def get_cognitive_efficiency(self) -> float:
        """
        Get cognitive efficiency based on adenosine/fatigue.

        Returns:
            Efficiency 0-1 (1=fully alert, 0=severely impaired)
        """
        if self.adenosine is None:
            return 1.0
        return self.adenosine.state.cognitive_efficiency

    def _compute_delay_term(self, fields: np.ndarray) -> np.ndarray:
        """
        Compute transmission delay contribution.

        Delays model axonal conduction and synaptic transmission
        times, creating more realistic temporal dynamics.

        Args:
            fields: Current NT fields [6, grid_size, ...]

        Returns:
            Delay contribution [6, grid_size, ...]
        """
        if self.delay_system is None:
            return np.zeros_like(fields)

        # Push current state to delay buffers
        self.delay_system.step(fields)

        # Get delayed NT fields (each NT has its own synaptic delay)
        delayed = self.delay_system.get_all_delayed_nts()

        # Delay feedback: difference between current and delayed
        # This creates temporal smoothing and realistic response times
        delay_strength = 0.05  # Moderate delay influence
        delay_term = delay_strength * (delayed - fields)

        return delay_term

    def get_transmission_delay(self, nt_index: int) -> float:
        """
        Get current transmission delay for a neurotransmitter.

        Args:
            nt_index: NT index [0=DA, 1=5HT, 2=ACh, 3=NE, 4=GABA, 5=Glu]

        Returns:
            Synaptic delay in milliseconds
        """
        if self.delay_system is None:
            return 0.0
        return self.delay_system.config.get_synaptic_delay(nt_index)

    def step(
        self,
        stimulus: np.ndarray | None = None,
        dt: float | None = None
    ) -> NeurotransmitterState:
        """
        Advance neural field by one time step.

        Implements semi-implicit Euler:
            U^(n+1) = U^n + dt * [-α*U^(n+1) + D∇²U^n + S^n + C(U^n)]

        Rearranging:
            U^(n+1) = (U^n + dt*[D∇²U^n + S^n + C(U^n)]) / (1 + dt*α)

        Args:
            stimulus: External input to NT fields [6, ...]
            dt: Override time step

        Returns:
            Spatially-averaged NeurotransmitterState
        """
        dt = dt or self._current_dt

        # Compute terms
        diffusion = self._compute_diffusion(self.fields)

        # P1-4: Check for NaN/Inf after diffusion
        if not np.all(np.isfinite(diffusion)):
            raise NumericalInstabilityError(
                f"NaN/Inf detected in diffusion term. "
                f"Non-finite count: {np.count_nonzero(~np.isfinite(diffusion))}"
            )

        coupling = self._compute_coupling_field(self.fields)
        attractor = self._compute_attractor_force(self.fields)
        da_ach = self._compute_da_ach_coupling(self.fields)
        astrocyte = self._compute_astrocyte_reuptake(self.fields)
        oscillation = self._compute_oscillation_drive(self.fields, dt)
        adenosine_mod = self._compute_adenosine_modulation(self.fields)
        delay_term = self._compute_delay_term(self.fields)
        lateral_inhib = self._compute_gaba_lateral_inhibition(self.fields)  # P1-3
        stim = stimulus if stimulus is not None else np.zeros_like(self.fields)

        # Explicit terms: D∇²U + S + C + DA-ACh + astrocyte + oscillations + adenosine + delays + lateral_inhib
        explicit_terms = diffusion + stim + coupling + attractor + da_ach + astrocyte + oscillation + adenosine_mod + delay_term + lateral_inhib

        # P1-4: Check for NaN/Inf after coupling
        if not np.all(np.isfinite(explicit_terms)):
            raise NumericalInstabilityError(
                f"NaN/Inf detected in explicit terms after coupling. "
                f"Non-finite count: {np.count_nonzero(~np.isfinite(explicit_terms))}"
            )

        # ATOM-P3-32: CFL condition check for reaction terms
        max_reaction_rate = np.max(np.abs(coupling + attractor))
        if dt * max_reaction_rate > 1.0:
            logger.warning(f"CFL violation in reaction: dt*rate={dt*max_reaction_rate:.3f} > 1.0")

        # Semi-implicit update: (U + dt*explicit) / (1 + dt*α)
        # This is unconditionally stable for decay term
        numerator = self.fields + dt * explicit_terms
        denominator = 1.0 + dt * self._alphas

        fields_new = numerator / denominator

        # P1-4: Check for NaN/Inf after integration step
        if not np.all(np.isfinite(fields_new)):
            raise NumericalInstabilityError(
                f"NaN/Inf detected in neural field state after integration. "
                f"Non-finite count: {np.count_nonzero(~np.isfinite(fields_new))}"
            )

        # Clamp to biological range
        fields_new = np.clip(
            fields_new,
            self.config.min_concentration,
            self.config.max_concentration
        )

        # Check for numerical stability
        if self.config.adaptive_timestepping:
            max_change = np.max(np.abs(fields_new - self.fields))

            # If change is too large, reject step and reduce dt
            if max_change > 0.5:  # More than 50% change in one step
                self._rejected_steps += 1
                self._current_dt = max(self._current_dt * 0.5, self.config.dt_min)
                logger.debug(
                    f"Step rejected (change={max_change:.3f}), "
                    f"reducing dt to {self._current_dt:.6f}"
                )
                # Retry with smaller dt
                return self.step(stimulus, self._current_dt)

            # If change is small, can increase dt
            elif max_change < 0.05:  # Less than 5% change
                self._current_dt = min(self._current_dt * 1.1, self.config.dt_max)

        # Accept step
        self.fields = fields_new
        self._time += dt
        self._step_count += 1

        # Update attractor state if manager exists
        if self.attractor_manager is not None:
            mean_state = self.get_mean_state()
            transition = self.attractor_manager.update(mean_state, dt)
            if transition is not None:
                logger.debug(
                    f"Cognitive state transition: "
                    f"{transition.from_state.name} -> {transition.to_state.name}"
                )

        return self.get_mean_state()

    def get_mean_state(self) -> NeurotransmitterState:
        """Get spatially-averaged NT state."""
        means = np.mean(self.fields, axis=tuple(range(1, self.fields.ndim)))
        return NeurotransmitterState.from_array(means)

    def get_field(self, nt_type: NeurotransmitterType) -> np.ndarray:
        """
        Get spatial field for specific neurotransmitter.

        Args:
            nt_type: Which neurotransmitter

        Returns:
            Spatial field [grid_size, ...]
        """
        nt_idx = nt_type.value - 1
        return self.fields[nt_idx].copy()

    def inject_stimulus(
        self,
        nt_type: NeurotransmitterType,
        magnitude: float,
        location: tuple | None = None,
        spatial_profile: np.ndarray | None = None
    ) -> None:
        """
        Inject external stimulus into specific NT field.

        Args:
            nt_type: Which neurotransmitter to stimulate
            magnitude: Stimulus strength [-1, 1]
            location: Spatial location (None = global)
            spatial_profile: Custom spatial pattern (overrides location)
        """
        nt_idx = nt_type.value - 1

        if spatial_profile is not None:
            # Use custom spatial pattern
            self.fields[nt_idx] += magnitude * spatial_profile
        elif location is None:
            # Global (uniform) stimulus
            self.fields[nt_idx] += magnitude
        else:
            # Point stimulus at specific location
            self.fields[nt_idx][location] += magnitude

        # Clamp to biological range
        self.fields = np.clip(
            self.fields,
            self.config.min_concentration,
            self.config.max_concentration
        )

    # -------------------------------------------------------------------------
    # Quick Win 3: DopamineSystem Integration
    # -------------------------------------------------------------------------

    def inject_rpe(
        self,
        rpe: float,
        magnitude_scale: float = 0.3,
        spatial_profile: np.ndarray | None = None
    ) -> None:
        """
        Inject reward prediction error (RPE) from DopamineSystem into DA field.

        Quick Win 3: Connects DopamineSystem output to neural field dynamics.

        Biological basis:
        - Positive RPE (reward > expected): DA burst (phasic increase)
        - Negative RPE (reward < expected): DA dip (phasic decrease)
        - Zero RPE (expected outcome): no change

        The RPE modulates the dopamine field, which then affects:
        - Learning in coupling matrix (via eligibility traces)
        - Attractor dynamics (via DA-dependent state transitions)
        - Memory consolidation priority

        Args:
            rpe: Reward prediction error from DopamineSystem [-1, 1]
            magnitude_scale: Scale factor for RPE -> DA field magnitude
            spatial_profile: Optional spatial pattern (None = global)
        """
        # Clamp RPE to valid range
        rpe_clamped = np.clip(rpe, -1.0, 1.0)

        # Convert RPE to DA field change
        # Positive RPE -> increase DA, negative RPE -> decrease DA
        da_change = rpe_clamped * magnitude_scale

        # Apply to DA field (index 0)
        if spatial_profile is not None:
            self.fields[0] += da_change * spatial_profile
        else:
            self.fields[0] += da_change

        # Clamp to biological range
        self.fields[0] = np.clip(
            self.fields[0],
            self.config.min_concentration,
            self.config.max_concentration
        )

        # If coupling exists, accumulate eligibility for later update
        if self.coupling is not None:
            current_state = self.get_mean_state()
            self.coupling.accumulate_eligibility(current_state)

        logger.debug(f"Injected RPE {rpe:.3f} -> DA change {da_change:.3f}")

    def inject_rpe_from_system(
        self,
        dopamine_system: DopamineSystem,
        memory_id: UUID,
        actual_outcome: float,
        update_expectations: bool = True
    ) -> float:
        """
        Compute and inject RPE using DopamineSystem, updating expectations.

        This is the main integration point between DopamineSystem and NeuralFieldSolver.

        Args:
            dopamine_system: The DopamineSystem instance
            memory_id: Memory that was retrieved/used
            actual_outcome: Observed outcome [0, 1]
            update_expectations: Whether to update DA system expectations

        Returns:
            The computed RPE value
        """


        # Compute RPE
        rpe_result = dopamine_system.compute_rpe(memory_id, actual_outcome)

        # Inject into neural field
        self.inject_rpe(rpe_result.rpe)

        # Update expectations if requested
        if update_expectations:
            dopamine_system.update_expectations(memory_id, actual_outcome)

        # If coupling exists, update with eligibility
        if self.coupling is not None and abs(rpe_result.rpe) > 0.05:
            current_state = self.get_mean_state()
            self.coupling.update_with_eligibility(
                current_state,
                rpe_result.rpe,
                use_accumulated=True
            )

        return rpe_result.rpe

    def get_da_rpe_modulated_learning_rate(
        self,
        base_lr: float,
        rpe: float
    ) -> float:
        """
        Compute surprise-modulated learning rate for other systems.

        Args:
            base_lr: Base learning rate
            rpe: Recent RPE value

        Returns:
            Modulated learning rate (higher for surprising outcomes)
        """
        surprise = abs(rpe)
        # Scale: |RPE|=0 -> 0.5x LR, |RPE|=1 -> 2x LR
        modulation = 0.5 + 1.5 * surprise
        return base_lr * modulation

    def inject_phasic_burst(
        self,
        nt_type: NeurotransmitterType,
        center: tuple | None = None,
        magnitude: float = 0.3,
        width: float = 3.0
    ) -> None:
        """
        Inject phasic burst (e.g., DA burst for reward).

        Creates Gaussian spatial profile centered at location.

        Args:
            nt_type: Which neurotransmitter
            center: Center location (None = center of grid)
            magnitude: Peak magnitude
            width: Spatial width (in grid points)
        """
        nt_idx = nt_type.value - 1

        # Default to center of grid
        if center is None:
            center = tuple([self.config.grid_size // 2] * self.config.spatial_dims)

        # Create Gaussian profile
        if self.config.spatial_dims == 1:
            x = np.arange(self.config.grid_size)
            profile = np.exp(-((x - center[0]) ** 2) / (2 * width ** 2))
        elif self.config.spatial_dims == 2:
            x = np.arange(self.config.grid_size)
            y = np.arange(self.config.grid_size)
            xx, yy = np.meshgrid(x, y)
            r_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
            profile = np.exp(-r_sq / (2 * width ** 2))
        elif self.config.spatial_dims == 3:
            x = np.arange(self.config.grid_size)
            y = np.arange(self.config.grid_size)
            z = np.arange(self.config.grid_size)
            xx, yy, zz = np.meshgrid(x, y, z)
            r_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
            profile = np.exp(-r_sq / (2 * width ** 2))
        else:
            raise ValueError(f"Unsupported spatial_dims: {self.config.spatial_dims}")

        self.inject_stimulus(nt_type, magnitude, spatial_profile=profile)

    def reset(self) -> None:
        """Reset fields to baseline."""
        self._init_fields()
        self._time = 0.0
        self._step_count = 0
        self._rejected_steps = 0
        self._current_dt = self.config.dt

    def get_stats(self) -> dict:
        """Get solver statistics."""
        state = self.get_mean_state()

        # Compute spatial statistics for each NT
        spatial_stats = {}
        for nt_type in NeurotransmitterType:
            field = self.get_field(nt_type)
            spatial_stats[nt_type.name.lower()] = {
                'mean': float(np.mean(field)),
                'std': float(np.std(field)),
                'min': float(np.min(field)),
                'max': float(np.max(field)),
            }

        return {
            'time': self._time,
            'step_count': self._step_count,
            'rejected_steps': self._rejected_steps,
            'current_dt': self._current_dt,
            'mean_state': state.to_array().tolist(),
            'spatial_stats': spatial_stats,
            'config': {
                'spatial_dims': self.config.spatial_dims,
                'grid_size': self.config.grid_size,
                'dt': self.config.dt,
                'adaptive': self.config.adaptive_timestepping,
            }
        }


# Type alias for external use
from ww.nca.coupling import LearnableCoupling  # noqa: E402

try:
    from ww.nca.attractors import StateTransitionManager  # noqa: E402
except ImportError:
    StateTransitionManager = None


__all__ = [
    "NeuralFieldConfig",
    "NeuralFieldSolver",
    "NeurotransmitterState",
    "NeurotransmitterType",
    "NumericalInstabilityError",
]
