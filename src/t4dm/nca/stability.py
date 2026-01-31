"""
Stability Analysis for NCA Energy Landscape.

Implements mathematically correct stability checks for energy-based dynamics:

1. Hessian Analysis:
   - Energy function E(U) has gradient ∇E and Hessian H = ∇²E
   - At equilibrium: ∇E = 0
   - Stable minimum: H positive definite (all eigenvalues > 0)
   - Saddle point: H indefinite (mixed sign eigenvalues)
   - Unstable maximum: H negative definite (all eigenvalues < 0)

2. Dynamics Stability:
   - For gradient descent: dU/dt = -∇E(U)
   - Jacobian of dynamics: J = -H
   - Stable dynamics: all eigenvalues of J have negative real parts
   - This is equivalent to H being positive definite

3. Lyapunov Analysis:
   - Energy E(U) serves as Lyapunov function
   - dE/dt = ∇E · dU/dt = -||∇E||² ≤ 0
   - System is Lyapunov stable if E decreases along trajectories

CRITICAL FIX (Hinton Audit):
The original Katie spec had a bug:
    return np.all(eigenvalues.real < 0)  # WRONG
This checked dynamics Jacobian eigenvalues, but on the Hessian directly.
Correct check depends on what matrix you're analyzing:
    - Hessian H: eigenvalues > 0 for stable minimum
    - Dynamics Jacobian J = -H: eigenvalues < 0 for stable dynamics

References:
- Strogatz: Nonlinear Dynamics and Chaos
- Hopfield (1984): Neurons with graded response
- Hinton Audit (2025): Katie stability analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from t4dm.nca.coupling import LearnableCoupling
    from t4dm.nca.energy import EnergyLandscape

logger = logging.getLogger(__name__)


class StabilityType(Enum):
    """Classification of equilibrium stability."""
    STABLE_NODE = auto()      # All eigenvalues negative real (asymptotically stable)
    STABLE_FOCUS = auto()     # Complex eigenvalues with negative real parts (spiral in)
    UNSTABLE_NODE = auto()    # All eigenvalues positive real
    UNSTABLE_FOCUS = auto()   # Complex eigenvalues with positive real parts (spiral out)
    SADDLE = auto()           # Mixed sign real parts
    CENTER = auto()           # Pure imaginary eigenvalues (marginally stable)
    DEGENERATE = auto()       # Zero eigenvalue (bifurcation point)


@dataclass
class StabilityResult:
    """Result of stability analysis at an equilibrium point."""

    # Equilibrium point analyzed
    equilibrium: np.ndarray

    # Eigenvalue analysis
    eigenvalues: np.ndarray  # Complex eigenvalues of Jacobian
    eigenvectors: np.ndarray  # Corresponding eigenvectors

    # Stability classification
    stability_type: StabilityType
    is_stable: bool  # True if asymptotically stable
    is_lyapunov_stable: bool  # True if Lyapunov stable (includes marginal)

    # Quantitative measures
    spectral_abscissa: float  # max(Re(eigenvalues)) - stability margin
    condition_number: float   # Sensitivity to perturbations
    lyapunov_exponent: float  # Largest Lyapunov exponent estimate

    # Energy landscape info
    energy_at_equilibrium: float
    gradient_norm: float  # Should be ~0 at true equilibrium
    hessian_eigenvalues: np.ndarray  # For energy curvature

    # Basin of attraction estimate
    basin_radius: float  # Estimated radius where linearization holds

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "equilibrium": self.equilibrium.tolist(),
            "eigenvalues_real": np.real(self.eigenvalues).tolist(),
            "eigenvalues_imag": np.imag(self.eigenvalues).tolist(),
            "stability_type": self.stability_type.name,
            "is_stable": self.is_stable,
            "is_lyapunov_stable": self.is_lyapunov_stable,
            "spectral_abscissa": self.spectral_abscissa,
            "condition_number": self.condition_number,
            "lyapunov_exponent": self.lyapunov_exponent,
            "energy_at_equilibrium": self.energy_at_equilibrium,
            "gradient_norm": self.gradient_norm,
            "hessian_eigenvalues": self.hessian_eigenvalues.tolist(),
            "basin_radius": self.basin_radius,
        }


@dataclass
class StabilityConfig:
    """Configuration for stability analysis."""

    # Numerical tolerances
    equilibrium_tol: float = 1e-6  # Gradient norm threshold for equilibrium
    eigenvalue_tol: float = 1e-10  # Threshold for zero eigenvalues

    # Hessian computation
    hessian_method: str = "analytical"  # "analytical" or "numerical"
    numerical_epsilon: float = 1e-5  # Step size for numerical derivatives

    # Basin estimation
    basin_probe_steps: int = 100  # Steps for basin radius estimation
    basin_probe_radius: float = 0.5  # Initial probe radius

    # Lyapunov analysis
    lyapunov_trajectory_length: int = 1000
    lyapunov_perturbation: float = 1e-8


class StabilityAnalyzer:
    """
    Stability analysis for NCA energy landscape.

    Provides mathematically rigorous stability analysis including:
    - Equilibrium detection and classification
    - Hessian-based curvature analysis
    - Jacobian eigenvalue analysis
    - Lyapunov stability verification
    - Basin of attraction estimation

    Usage:
        analyzer = StabilityAnalyzer(energy_landscape)

        # Analyze stability at a point
        result = analyzer.analyze_equilibrium(U_eq)

        # Check if point is stable
        if result.is_stable:
            logger.info("Stable equilibrium")

        # Find all attractors
        attractors = analyzer.find_attractors(n_samples=100)
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape | None = None,
        coupling: LearnableCoupling | None = None,
        config: StabilityConfig | None = None,
    ):
        """
        Initialize stability analyzer.

        Args:
            energy_landscape: Energy landscape to analyze
            coupling: Coupling matrix (used if no energy_landscape)
            config: Analysis configuration
        """
        self.energy_landscape = energy_landscape
        self.coupling = coupling
        self.config = config or StabilityConfig()

        # Validate we have something to analyze
        if energy_landscape is None and coupling is None:
            raise ValueError("Must provide either energy_landscape or coupling")

        logger.info("StabilityAnalyzer initialized")

    # -------------------------------------------------------------------------
    # Hessian Computation
    # -------------------------------------------------------------------------

    def compute_hessian(
        self,
        U: np.ndarray,
        method: str | None = None
    ) -> np.ndarray:
        """
        Compute Hessian matrix H = ∇²E at point U.

        For Hopfield energy E = -0.5 * U^T K U + boundary + attractor:
            H_ij = ∂²E/∂U_i∂U_j

        Args:
            U: Point to compute Hessian at [6]
            method: "analytical" or "numerical"

        Returns:
            Hessian matrix [6, 6]
        """
        method = method or self.config.hessian_method

        if method == "analytical":
            return self._compute_hessian_analytical(U)
        else:
            return self._compute_hessian_numerical(U)

    def _compute_hessian_analytical(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Hessian analytically.

        For Hopfield: E = -0.5 * U^T K U
            ∇E = -K @ U (for symmetric K)
            H = -K

        For boundary energy (softplus):
            H_ii = p * s² * [σ(-sU_i)(1-σ(-sU_i)) + σ(s(U_i-1))(1-σ(s(U_i-1)))]

        For attractor energy (Gaussian wells):
            H is more complex, computed numerically if needed
        """
        n = len(U)
        H = np.zeros((n, n), dtype=np.float64)

        # Hopfield term: H = -K (for symmetric K)
        if self.coupling is not None:
            K = self.coupling.K
            K_sym = 0.5 * (K + K.T)
            # Energy is -0.5 * U^T K U, so gradient is -K @ U
            # Hessian is -K
            if self.energy_landscape is not None:
                scale = self.energy_landscape.config.hopfield_scale
            else:
                scale = 1.0
            H -= scale * K_sym

        # Boundary term (diagonal)
        if self.energy_landscape is not None:
            s = self.energy_landscape.config.boundary_steepness
            p = self.energy_landscape.config.boundary_penalty

            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

            # Second derivative of softplus: s * sigmoid(x) * (1 - sigmoid(x))
            for i in range(n):
                sig_low = sigmoid(-s * U[i])
                sig_high = sigmoid(s * (U[i] - 1))

                # d²/dU² of softplus terms
                H[i, i] += p * s * (sig_low * (1 - sig_low) + sig_high * (1 - sig_high))

        # Attractor term - use numerical for complexity
        if self.energy_landscape is not None and self.energy_landscape.state_manager is not None:
            H_attr = self._compute_attractor_hessian_numerical(U)
            H += H_attr

        return H

    def _compute_hessian_numerical(self, U: np.ndarray) -> np.ndarray:
        """Compute Hessian via central differences."""
        n = len(U)
        base_eps = self.config.numerical_epsilon
        H = np.zeros((n, n), dtype=np.float64)

        def grad(u):
            if self.energy_landscape is not None:
                return self.energy_landscape.compute_energy_gradient(u)
            else:
                # Simple Hopfield gradient
                K = self.coupling.K
                K_sym = 0.5 * (K + K.T)
                return -K_sym @ u

        for i in range(n):
            eps = base_eps * max(1.0, abs(U[i]))
            U_plus = U.copy()
            U_minus = U.copy()
            U_plus[i] += eps
            U_minus[i] -= eps

            grad_plus = grad(U_plus)
            grad_minus = grad(U_minus)

            H[:, i] = (grad_plus - grad_minus) / (2 * eps)

        # Symmetrize (should be symmetric, but numerical errors)
        H = 0.5 * (H + H.T)

        return H

    def _compute_attractor_hessian_numerical(self, U: np.ndarray) -> np.ndarray:
        """Compute Hessian of attractor energy numerically."""
        n = len(U)
        eps = self.config.numerical_epsilon
        H = np.zeros((n, n), dtype=np.float64)

        def attr_grad(u):
            return self.energy_landscape.compute_attractor_gradient(u)

        for i in range(n):
            U_plus = U.copy()
            U_minus = U.copy()
            U_plus[i] += eps
            U_minus[i] -= eps

            grad_plus = attr_grad(U_plus)
            grad_minus = attr_grad(U_minus)

            H[:, i] = (grad_plus - grad_minus) / (2 * eps)

        return 0.5 * (H + H.T)

    # -------------------------------------------------------------------------
    # Eigenvalue Analysis
    # -------------------------------------------------------------------------

    def compute_dynamics_jacobian(self, U: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of dynamics dU/dt = -∇E(U).

        J = -H where H is the Hessian.

        For stable dynamics, eigenvalues of J should have negative real parts.

        Args:
            U: Point to compute Jacobian at

        Returns:
            Jacobian matrix [6, 6]
        """
        H = self.compute_hessian(U)
        return -H

    def analyze_eigenvalues(
        self,
        matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, StabilityType]:
        """
        Analyze eigenvalues and classify stability.

        Args:
            matrix: Matrix to analyze (Jacobian for dynamics stability)

        Returns:
            (eigenvalues, eigenvectors, stability_type)
        """
        eigenvalues, eigenvectors = linalg.eig(matrix)

        # Classify based on eigenvalue real parts
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        tol = self.config.eigenvalue_tol

        has_positive = np.any(real_parts > tol)
        has_negative = np.any(real_parts < -tol)
        has_zero = np.any(np.abs(real_parts) <= tol)
        has_imaginary = np.any(np.abs(imag_parts) > tol)

        if has_zero:
            stability_type = StabilityType.DEGENERATE
        elif has_positive and has_negative:
            stability_type = StabilityType.SADDLE
        elif has_positive:
            if has_imaginary:
                stability_type = StabilityType.UNSTABLE_FOCUS
            else:
                stability_type = StabilityType.UNSTABLE_NODE
        elif has_negative:
            if has_imaginary:
                stability_type = StabilityType.STABLE_FOCUS
            else:
                stability_type = StabilityType.STABLE_NODE
        else:
            # All zero real parts with imaginary parts
            stability_type = StabilityType.CENTER

        return eigenvalues, eigenvectors, stability_type

    def is_positive_definite(self, matrix: np.ndarray) -> bool:
        """
        Check if matrix is positive definite.

        A matrix is positive definite if all eigenvalues are positive.
        This is the CORRECT check for a stable energy minimum.

        Args:
            matrix: Symmetric matrix to check

        Returns:
            True if positive definite
        """
        eigenvalues = linalg.eigvalsh(matrix)  # Real eigenvalues for symmetric
        return np.all(eigenvalues > self.config.eigenvalue_tol)

    def is_stable_equilibrium(self, U: np.ndarray) -> bool:
        """
        Check if U is a stable equilibrium of the energy landscape.

        CORRECT IMPLEMENTATION:
        1. Check gradient is near zero (equilibrium condition)
        2. Check Hessian is positive definite (stable minimum)

        This fixes the bug in the original Katie spec that used wrong sign.

        Args:
            U: Point to check

        Returns:
            True if stable equilibrium
        """
        # Check equilibrium condition
        if self.energy_landscape is not None:
            grad = self.energy_landscape.compute_energy_gradient(U)
        else:
            K = self.coupling.K
            K_sym = 0.5 * (K + K.T)
            grad = -K_sym @ U

        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.config.equilibrium_tol:
            logger.debug(f"Not equilibrium: gradient norm = {grad_norm}")
            return False

        # Check Hessian positive definiteness
        H = self.compute_hessian(U)
        is_pd = self.is_positive_definite(H)

        if not is_pd:
            eigenvalues = linalg.eigvalsh(H)
            logger.debug(f"Not stable: Hessian eigenvalues = {eigenvalues}")

        return is_pd

    # -------------------------------------------------------------------------
    # Full Stability Analysis
    # -------------------------------------------------------------------------

    def analyze_equilibrium(self, U: np.ndarray) -> StabilityResult:
        """
        Perform complete stability analysis at point U.

        Args:
            U: Point to analyze

        Returns:
            StabilityResult with complete analysis
        """
        U = np.asarray(U, dtype=np.float64)

        # Compute gradient
        if self.energy_landscape is not None:
            grad = self.energy_landscape.compute_energy_gradient(U)
            energy = self.energy_landscape.compute_total_energy(U)
        else:
            K = self.coupling.K
            K_sym = 0.5 * (K + K.T)
            grad = -K_sym @ U
            energy = -0.5 * float(U @ K_sym @ U)

        grad_norm = float(np.linalg.norm(grad))

        # Compute Hessian and its eigenvalues
        H = self.compute_hessian(U)
        hessian_eigenvalues = linalg.eigvalsh(H)

        # Compute dynamics Jacobian and its eigenvalues
        J = -H  # J = -H for gradient descent dynamics
        eigenvalues, eigenvectors, stability_type = self.analyze_eigenvalues(J)

        # Stability determination
        # For dynamics dU/dt = -∇E, stable if all eigenvalues of J have Re < 0
        # This is equivalent to H being positive definite
        spectral_abscissa = float(np.max(np.real(eigenvalues)))
        is_stable = spectral_abscissa < -self.config.eigenvalue_tol
        is_lyapunov_stable = spectral_abscissa <= self.config.eigenvalue_tol

        # Condition number (for sensitivity analysis)
        try:
            condition_number = float(np.linalg.cond(H))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            condition_number = float('inf')

        # Lyapunov exponent (approximated by spectral abscissa for linear)
        lyapunov_exponent = spectral_abscissa

        # Basin radius estimation
        basin_radius = self._estimate_basin_radius(U, H)

        return StabilityResult(
            equilibrium=U.copy(),
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            stability_type=stability_type,
            is_stable=is_stable,
            is_lyapunov_stable=is_lyapunov_stable,
            spectral_abscissa=spectral_abscissa,
            condition_number=condition_number,
            lyapunov_exponent=lyapunov_exponent,
            energy_at_equilibrium=energy,
            gradient_norm=grad_norm,
            hessian_eigenvalues=hessian_eigenvalues,
            basin_radius=basin_radius,
        )

    def _estimate_basin_radius(
        self,
        U_eq: np.ndarray,
        H: np.ndarray
    ) -> float:
        """
        Estimate radius of basin of attraction.

        Uses linearization validity: basin extends until nonlinear
        terms become significant.

        Args:
            U_eq: Equilibrium point
            H: Hessian at equilibrium

        Returns:
            Estimated basin radius
        """
        if self.energy_landscape is None:
            # Simple estimate based on eigenvalue magnitude
            eigenvalues = linalg.eigvalsh(H)
            if np.all(eigenvalues > 0):
                # Stable: basin size inversely proportional to curvature
                return float(1.0 / np.sqrt(np.max(eigenvalues)))
            else:
                return 0.0

        # More sophisticated: probe until energy increases
        min_eigenvalue = np.min(linalg.eigvalsh(H))
        if min_eigenvalue <= 0:
            return 0.0  # Not a minimum

        # Sample directions and find where we leave the basin
        n_samples = 20
        radii = []

        for _ in range(n_samples):
            direction = np.random.randn(len(U_eq))
            direction = direction / np.linalg.norm(direction)

            # Binary search for basin edge
            r_low, r_high = 0.0, self.config.basin_probe_radius
            E_eq = self.energy_landscape.compute_total_energy(U_eq)

            for _ in range(10):
                r_mid = (r_low + r_high) / 2
                U_probe = U_eq + r_mid * direction
                U_probe = np.clip(U_probe, 0, 1)

                # Check if we can return to equilibrium
                U_relaxed, _, E_final = self.energy_landscape.relax_to_equilibrium(
                    U_probe, max_steps=self.config.basin_probe_steps
                )

                if np.linalg.norm(U_relaxed - U_eq) < 0.1:
                    r_low = r_mid
                else:
                    r_high = r_mid

            radii.append(r_low)

        return float(np.mean(radii)) if radii else 0.0

    # -------------------------------------------------------------------------
    # Attractor Finding
    # -------------------------------------------------------------------------

    def find_attractors(
        self,
        n_samples: int = 100,
        bounds: tuple[float, float] = (0.0, 1.0)
    ) -> list[StabilityResult]:
        """
        Find stable attractors by sampling and relaxing.

        Args:
            n_samples: Number of random initial conditions
            bounds: Range for initial conditions

        Returns:
            List of unique stable attractors
        """
        if self.energy_landscape is None:
            raise ValueError("Need energy_landscape to find attractors")

        attractors = []
        seen_equilibria = []

        for _ in range(n_samples):
            # Random initial condition
            U0 = np.random.uniform(bounds[0], bounds[1], 6).astype(np.float32)

            # Relax to equilibrium
            U_eq, steps, energy = self.energy_landscape.relax_to_equilibrium(
                U0, max_steps=500, use_langevin=False
            )

            # Check if this is a new equilibrium
            is_new = True
            for seen in seen_equilibria:
                if np.linalg.norm(U_eq - seen) < 0.05:
                    is_new = False
                    break

            if is_new:
                result = self.analyze_equilibrium(U_eq)
                if result.is_stable:
                    attractors.append(result)
                    seen_equilibria.append(U_eq)

        return attractors

    # -------------------------------------------------------------------------
    # Lyapunov Analysis
    # -------------------------------------------------------------------------

    def compute_lyapunov_exponent(
        self,
        U0: np.ndarray,
        trajectory_length: int | None = None
    ) -> float:
        """
        Compute largest Lyapunov exponent via trajectory divergence.

        ATOM-P4-12: Simplified Lyapunov estimation via trajectory divergence.
        For publication-grade spectra, implement Benettin et al. 1980.

        The Lyapunov exponent measures the rate of separation of
        infinitesimally close trajectories.

        λ > 0: Chaos (exponential divergence)
        λ < 0: Stable (trajectories converge)
        λ ≈ 0: Marginal stability

        Args:
            U0: Initial condition
            trajectory_length: Number of steps

        Returns:
            Estimated largest Lyapunov exponent
        """
        if self.energy_landscape is None:
            raise ValueError("Need energy_landscape for Lyapunov computation")

        length = trajectory_length or self.config.lyapunov_trajectory_length
        eps = self.config.lyapunov_perturbation

        U = U0.copy()
        delta = np.random.randn(len(U))
        delta = delta / np.linalg.norm(delta) * eps
        U_perturbed = U + delta

        lyapunov_sum = 0.0

        for i in range(length):
            # Evolve both trajectories
            U = self.energy_landscape.gradient_step(U, lr=0.01)
            U_perturbed = self.energy_landscape.gradient_step(U_perturbed, lr=0.01)

            # Compute separation
            separation = np.linalg.norm(U_perturbed - U)

            if separation > 0:
                lyapunov_sum += np.log(separation / eps)

                # Renormalize perturbation
                delta = (U_perturbed - U) / separation * eps
                U_perturbed = U + delta

        return float(lyapunov_sum / length)

    def verify_lyapunov_stability(
        self,
        U_eq: np.ndarray,
        n_perturbations: int = 20,
        perturbation_radius: float = 0.1
    ) -> dict:
        """
        Verify Lyapunov stability by checking energy decrease.

        For gradient descent on energy E:
            dE/dt = ∇E · dU/dt = -||∇E||² ≤ 0

        Energy should monotonically decrease along trajectories.

        Args:
            U_eq: Equilibrium to test
            n_perturbations: Number of random perturbations
            perturbation_radius: Size of perturbations

        Returns:
            Verification results
        """
        if self.energy_landscape is None:
            raise ValueError("Need energy_landscape for Lyapunov verification")

        E_eq = self.energy_landscape.compute_total_energy(U_eq)
        results = {
            "equilibrium_energy": E_eq,
            "perturbations_tested": n_perturbations,
            "all_converged": True,
            "energy_always_decreased": True,
            "trajectories": [],
        }

        for i in range(n_perturbations):
            # Random perturbation
            delta = np.random.randn(len(U_eq))
            delta = delta / np.linalg.norm(delta) * perturbation_radius
            U = U_eq + delta
            U = np.clip(U, 0, 1)

            # Track trajectory
            trajectory = {"energies": [], "distances": []}
            prev_energy = self.energy_landscape.compute_total_energy(U)

            for step in range(100):
                U = self.energy_landscape.gradient_step(U, lr=0.01)
                energy = self.energy_landscape.compute_total_energy(U)
                distance = float(np.linalg.norm(U - U_eq))

                trajectory["energies"].append(energy)
                trajectory["distances"].append(distance)

                # Check energy decrease
                if energy > prev_energy + 1e-10:  # Small tolerance
                    results["energy_always_decreased"] = False

                prev_energy = energy

            # Check convergence
            final_distance = trajectory["distances"][-1]
            if final_distance > 0.01:  # Didn't converge
                results["all_converged"] = False

            results["trajectories"].append(trajectory)

        results["is_lyapunov_stable"] = (
            results["all_converged"] and results["energy_always_decreased"]
        )

        return results

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_stability_summary(self, U: np.ndarray) -> str:
        """Get human-readable stability summary."""
        result = self.analyze_equilibrium(U)

        summary = [
            f"Stability Analysis at U = {U}",
            f"  Type: {result.stability_type.name}",
            f"  Stable: {result.is_stable}",
            f"  Lyapunov Stable: {result.is_lyapunov_stable}",
            f"  Spectral Abscissa: {result.spectral_abscissa:.6f}",
            f"  Energy: {result.energy_at_equilibrium:.4f}",
            f"  Gradient Norm: {result.gradient_norm:.2e}",
            f"  Hessian Eigenvalues: {result.hessian_eigenvalues}",
            f"  Basin Radius: {result.basin_radius:.4f}",
        ]

        return "\n".join(summary)


def check_energy_stability(
    coupling: LearnableCoupling,
    U: np.ndarray
) -> bool:
    """
    Quick stability check for coupling matrix at point U.

    CORRECT IMPLEMENTATION (fixes Katie bug):
    For energy E = -0.5 * U^T K U, stable minimum requires:
    - Hessian H = -K to be positive definite
    - Equivalently, K should be negative definite

    Args:
        coupling: Coupling matrix
        U: Point to check

    Returns:
        True if stable minimum
    """
    K = coupling.K
    K_sym = 0.5 * (K + K.T)
    H = -K_sym  # Hessian of Hopfield energy

    eigenvalues = linalg.eigvalsh(H)

    # Positive definite = stable minimum
    # NOT eigenvalues.real < 0 (that was the bug!)
    return bool(np.all(eigenvalues > 1e-10))


__all__ = [
    "StabilityAnalyzer",
    "StabilityConfig",
    "StabilityResult",
    "StabilityType",
    "check_energy_stability",
]
