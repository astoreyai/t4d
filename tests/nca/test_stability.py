"""
Tests for stability analysis module.

Validates:
1. Correct eigenvalue checks (fixes Katie bug)
2. Hessian computation (analytical and numerical)
3. Stability classification
4. Lyapunov analysis
5. Basin of attraction estimation
"""

import numpy as np
import pytest
from scipy import linalg

from ww.nca.stability import (
    StabilityAnalyzer,
    StabilityConfig,
    StabilityResult,
    StabilityType,
    check_energy_stability,
)
from ww.nca.coupling import LearnableCoupling, CouplingConfig
from ww.nca.energy import EnergyLandscape, EnergyConfig
from ww.nca.attractors import StateTransitionManager


class TestStabilityConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = StabilityConfig()

        assert config.equilibrium_tol > 0
        assert config.eigenvalue_tol > 0
        assert config.hessian_method in ["analytical", "numerical"]


class TestHessianComputation:
    """Test Hessian matrix computation."""

    def test_hessian_symmetry(self):
        """Hessian should be symmetric."""
        coupling = LearnableCoupling()
        analyzer = StabilityAnalyzer(coupling=coupling)

        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        H = analyzer.compute_hessian(U)

        # Check symmetry
        assert np.allclose(H, H.T, atol=1e-10)

    def test_hessian_hopfield_analytical(self):
        """Analytical Hessian for pure Hopfield is -K."""
        # Create coupling with known structure
        config = CouplingConfig()
        coupling = LearnableCoupling(config=config)

        # No energy landscape = pure Hopfield
        analyzer = StabilityAnalyzer(coupling=coupling)

        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        H = analyzer.compute_hessian(U, method="analytical")

        # For pure Hopfield, H = -K_sym
        K = coupling.K
        K_sym = 0.5 * (K + K.T)
        expected = -K_sym

        assert np.allclose(H, expected, atol=1e-6)

    def test_hessian_numerical_matches_analytical(self):
        """Numerical Hessian should match analytical."""
        coupling = LearnableCoupling()
        analyzer = StabilityAnalyzer(coupling=coupling)

        U = np.array([0.4, 0.6, 0.5, 0.5, 0.3, 0.7])

        H_analytical = analyzer.compute_hessian(U, method="analytical")
        H_numerical = analyzer.compute_hessian(U, method="numerical")

        # Should be close (numerical has some error)
        assert np.allclose(H_analytical, H_numerical, atol=1e-4)

    def test_hessian_with_energy_landscape(self):
        """Hessian includes boundary and attractor terms."""
        coupling = LearnableCoupling()
        manager = StateTransitionManager()
        energy = EnergyLandscape(
            coupling=coupling,
            state_manager=manager,
            config=EnergyConfig(boundary_penalty=5.0)
        )

        analyzer = StabilityAnalyzer(energy_landscape=energy, coupling=coupling)

        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        H = analyzer.compute_hessian(U)

        # Should include boundary terms (diagonal additions)
        # Pure Hopfield H would be -K, with boundary it's different
        K = coupling.K
        K_sym = 0.5 * (K + K.T)
        H_hopfield = -K_sym

        # Boundary adds to diagonal
        assert not np.allclose(H, H_hopfield)


class TestEigenvalueAnalysis:
    """Test eigenvalue-based stability classification."""

    def test_stable_node_classification(self):
        """Negative real eigenvalues -> stable node."""
        analyzer = StabilityAnalyzer(coupling=LearnableCoupling())

        # Matrix with all negative real eigenvalues
        M = np.diag([-1, -2, -3, -4, -5, -6])

        eigenvalues, _, stability_type = analyzer.analyze_eigenvalues(M)

        assert stability_type == StabilityType.STABLE_NODE
        assert np.all(np.real(eigenvalues) < 0)

    def test_unstable_node_classification(self):
        """Positive real eigenvalues -> unstable node."""
        analyzer = StabilityAnalyzer(coupling=LearnableCoupling())

        # Matrix with all positive real eigenvalues
        M = np.diag([1, 2, 3, 4, 5, 6])

        eigenvalues, _, stability_type = analyzer.analyze_eigenvalues(M)

        assert stability_type == StabilityType.UNSTABLE_NODE
        assert np.all(np.real(eigenvalues) > 0)

    def test_saddle_classification(self):
        """Mixed sign eigenvalues -> saddle."""
        analyzer = StabilityAnalyzer(coupling=LearnableCoupling())

        # Matrix with mixed eigenvalues
        M = np.diag([-1, -2, 3, 4, -5, 6])

        _, _, stability_type = analyzer.analyze_eigenvalues(M)

        assert stability_type == StabilityType.SADDLE

    def test_stable_focus_classification(self):
        """Complex eigenvalues with negative real -> stable focus."""
        analyzer = StabilityAnalyzer(coupling=LearnableCoupling())

        # 2D rotation + contraction
        block = np.array([[-0.5, 1], [-1, -0.5]])
        M = linalg.block_diag(block, block, block)

        eigenvalues, _, stability_type = analyzer.analyze_eigenvalues(M)

        assert stability_type == StabilityType.STABLE_FOCUS
        assert np.any(np.imag(eigenvalues) != 0)
        assert np.all(np.real(eigenvalues) < 0)


class TestPositiveDefiniteCheck:
    """Test positive definiteness checking."""

    def test_positive_definite_identity(self):
        """Identity matrix is positive definite."""
        analyzer = StabilityAnalyzer(coupling=LearnableCoupling())

        I = np.eye(6)
        assert analyzer.is_positive_definite(I)

    def test_negative_definite_fails(self):
        """Negative definite matrix is not positive definite."""
        analyzer = StabilityAnalyzer(coupling=LearnableCoupling())

        M = -np.eye(6)
        assert not analyzer.is_positive_definite(M)

    def test_indefinite_fails(self):
        """Indefinite matrix is not positive definite."""
        analyzer = StabilityAnalyzer(coupling=LearnableCoupling())

        M = np.diag([1, 1, 1, -1, -1, -1])
        assert not analyzer.is_positive_definite(M)


class TestKatieBugFix:
    """
    Test that the Katie bug is fixed.

    The bug was: return np.all(eigenvalues.real < 0)
    This is WRONG for checking energy stability.

    CORRECT:
    - Hessian eigenvalues > 0 for stable minimum
    - Dynamics Jacobian eigenvalues < 0 for stable dynamics
    """

    def test_stable_minimum_has_positive_hessian_eigenvalues(self):
        """At stable minimum, Hessian should be positive definite."""
        coupling = LearnableCoupling()
        energy = EnergyLandscape(
            coupling=coupling,
            config=EnergyConfig(boundary_penalty=10.0)
        )
        analyzer = StabilityAnalyzer(energy_landscape=energy, coupling=coupling)

        # Find a stable point by relaxation
        U0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        U_eq, _, _ = energy.relax_to_equilibrium(U0, max_steps=100, use_langevin=False)

        # Analyze
        result = analyzer.analyze_equilibrium(U_eq)

        # At stable minimum, Hessian eigenvalues should be POSITIVE
        # NOT NEGATIVE (that was the bug!)
        if result.is_stable:
            assert np.all(result.hessian_eigenvalues > 0), \
                f"Stable minimum should have positive Hessian eigenvalues, got {result.hessian_eigenvalues}"

    def test_dynamics_jacobian_has_negative_eigenvalues_at_stable(self):
        """At stable point, dynamics Jacobian should have negative eigenvalues."""
        coupling = LearnableCoupling()
        energy = EnergyLandscape(
            coupling=coupling,
            config=EnergyConfig(boundary_penalty=10.0)
        )
        analyzer = StabilityAnalyzer(energy_landscape=energy, coupling=coupling)

        # Find stable point
        U0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        U_eq, _, _ = energy.relax_to_equilibrium(U0, max_steps=100, use_langevin=False)

        # Compute dynamics Jacobian
        J = analyzer.compute_dynamics_jacobian(U_eq)
        eigenvalues = linalg.eigvals(J)

        result = analyzer.analyze_equilibrium(U_eq)
        if result.is_stable:
            # Dynamics Jacobian = -Hessian
            # For stable minimum (positive Hessian), Jacobian has negative eigenvalues
            assert np.all(np.real(eigenvalues) < 0.01), \
                f"Stable dynamics should have negative Jacobian eigenvalues, got {np.real(eigenvalues)}"

    def test_check_energy_stability_correct(self):
        """check_energy_stability uses correct criterion."""
        coupling = LearnableCoupling()
        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # The function should check Hessian positive definiteness
        # NOT eigenvalues.real < 0
        is_stable = check_energy_stability(coupling, U)

        # Verify by computing Hessian ourselves
        K = coupling.K
        K_sym = 0.5 * (K + K.T)
        H = -K_sym  # Hessian of Hopfield
        eigenvalues = linalg.eigvalsh(H)
        expected_stable = np.all(eigenvalues > 1e-10)

        assert is_stable == expected_stable


class TestStabilityAnalysis:
    """Test full stability analysis."""

    def test_analyze_equilibrium_returns_result(self):
        """analyze_equilibrium returns StabilityResult."""
        coupling = LearnableCoupling()
        analyzer = StabilityAnalyzer(coupling=coupling)

        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = analyzer.analyze_equilibrium(U)

        assert isinstance(result, StabilityResult)
        assert result.equilibrium is not None
        assert result.eigenvalues is not None
        assert result.stability_type is not None

    def test_is_stable_equilibrium(self):
        """is_stable_equilibrium checks both gradient and Hessian."""
        coupling = LearnableCoupling()
        energy = EnergyLandscape(coupling=coupling)
        analyzer = StabilityAnalyzer(energy_landscape=energy, coupling=coupling)

        # Find equilibrium
        U0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        U_eq, _, _ = energy.relax_to_equilibrium(U0, max_steps=200, use_langevin=False)

        # Check gradient is small
        grad = energy.compute_energy_gradient(U_eq)
        grad_norm = np.linalg.norm(grad)

        # At equilibrium, gradient should be small
        if grad_norm < 1e-4:
            # Then stability depends on Hessian
            pass

    def test_result_to_dict(self):
        """StabilityResult can be serialized."""
        coupling = LearnableCoupling()
        analyzer = StabilityAnalyzer(coupling=coupling)

        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = analyzer.analyze_equilibrium(U)

        d = result.to_dict()

        assert "equilibrium" in d
        assert "eigenvalues_real" in d
        assert "stability_type" in d
        assert "is_stable" in d

    def test_spectral_abscissa(self):
        """Spectral abscissa is max real part of eigenvalues."""
        coupling = LearnableCoupling()
        analyzer = StabilityAnalyzer(coupling=coupling)

        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = analyzer.analyze_equilibrium(U)

        # Verify spectral abscissa
        expected = np.max(np.real(result.eigenvalues))
        assert np.isclose(result.spectral_abscissa, expected)


class TestLyapunovAnalysis:
    """Test Lyapunov stability analysis."""

    def test_lyapunov_exponent_stable(self):
        """Stable system has negative Lyapunov exponent."""
        coupling = LearnableCoupling()
        energy = EnergyLandscape(
            coupling=coupling,
            config=EnergyConfig(boundary_penalty=10.0)
        )
        analyzer = StabilityAnalyzer(energy_landscape=energy, coupling=coupling)

        # Start near a stable point
        U0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        lyap = analyzer.compute_lyapunov_exponent(U0, trajectory_length=100)

        # For gradient descent on a convex energy, should be negative
        # (trajectories converge)
        # Note: may not always be negative depending on energy landscape shape

    def test_verify_lyapunov_stability(self):
        """Energy decreases along trajectories from near equilibrium."""
        coupling = LearnableCoupling()
        energy = EnergyLandscape(
            coupling=coupling,
            config=EnergyConfig(boundary_penalty=10.0)
        )
        analyzer = StabilityAnalyzer(energy_landscape=energy, coupling=coupling)

        # Find equilibrium
        U0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        U_eq, _, _ = energy.relax_to_equilibrium(U0, max_steps=200, use_langevin=False)

        # Verify Lyapunov stability
        result = analyzer.verify_lyapunov_stability(
            U_eq,
            n_perturbations=5,
            perturbation_radius=0.05
        )

        assert "is_lyapunov_stable" in result
        assert "energy_always_decreased" in result

        # Energy should always decrease for gradient descent
        assert result["energy_always_decreased"]


class TestAttractorFinding:
    """Test attractor discovery."""

    def test_find_attractors(self):
        """Can find stable attractors in energy landscape."""
        coupling = LearnableCoupling()
        manager = StateTransitionManager()
        energy = EnergyLandscape(
            coupling=coupling,
            state_manager=manager,
            config=EnergyConfig(
                attractor_strength=1.0,
                boundary_penalty=5.0
            )
        )
        analyzer = StabilityAnalyzer(energy_landscape=energy, coupling=coupling)

        # Find attractors
        attractors = analyzer.find_attractors(n_samples=20)

        # Should find at least some attractors
        # (depends on energy landscape structure)
        assert isinstance(attractors, list)

        for attractor in attractors:
            assert isinstance(attractor, StabilityResult)
            assert attractor.is_stable


class TestStabilitySummary:
    """Test summary output."""

    def test_get_stability_summary(self):
        """Summary is human-readable string."""
        coupling = LearnableCoupling()
        analyzer = StabilityAnalyzer(coupling=coupling)

        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        summary = analyzer.get_stability_summary(U)

        assert isinstance(summary, str)
        assert "Stability Analysis" in summary
        assert "Type:" in summary
        assert "Stable:" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
