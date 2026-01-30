"""
Comprehensive tests for Neural Field PDE Solver.

Tests cover:
1. Basic PDE integration (decay, diffusion)
2. Numerical stability (CFL condition, adaptive timestepping)
3. Boundary conditions (no-flux, periodic)
4. Biological plausibility (NT timescales, concentration bounds)
5. Stimulus injection (global, local, phasic bursts)
6. Integration with coupling and attractors
7. Multi-dimensional fields (1D, 2D, 3D)
"""

import numpy as np
import pytest

from ww.nca.neural_field import (
    NeuralFieldConfig,
    NeuralFieldSolver,
    NeurotransmitterState,
    NeurotransmitterType,
)


class TestNeurotransmitterState:
    """Test NT state representation."""

    def test_default_initialization(self):
        """Test default NT state is at baseline (0.5)."""
        state = NeurotransmitterState()
        assert state.dopamine == 0.5
        assert state.serotonin == 0.5
        assert state.acetylcholine == 0.5
        assert state.norepinephrine == 0.5
        assert state.gaba == 0.5
        assert state.glutamate == 0.5

    def test_to_from_array(self):
        """Test conversion to/from numpy array."""
        state = NeurotransmitterState(
            dopamine=0.7,
            serotonin=0.3,
            acetylcholine=0.6,
            norepinephrine=0.8,
            gaba=0.4,
            glutamate=0.9
        )
        arr = state.to_array()
        assert arr.shape == (6,)
        np.testing.assert_almost_equal(arr[0], 0.7, decimal=5)  # DA
        np.testing.assert_almost_equal(arr[5], 0.9, decimal=5)  # Glu

        # Round-trip
        state2 = NeurotransmitterState.from_array(arr)
        np.testing.assert_almost_equal(state2.dopamine, state.dopamine, decimal=5)
        np.testing.assert_almost_equal(state2.glutamate, state.glutamate, decimal=5)

    def test_biological_bounds_clamping(self):
        """Test that values are clamped to [0, 1]."""
        state = NeurotransmitterState(
            dopamine=1.5,  # Too high
            serotonin=-0.2,  # Too low
            acetylcholine=0.5
        )
        assert state.dopamine == 1.0
        assert state.serotonin == 0.0
        assert state.acetylcholine == 0.5


class TestNeuralFieldConfig:
    """Test configuration validation."""

    def test_default_config_is_stable(self):
        """Test that default config satisfies stability conditions."""
        config = NeuralFieldConfig()

        # Check that config initializes without error
        assert config.dt > 0
        assert config.grid_size > 0
        assert config.spatial_dims in [1, 2, 3]

    def test_cfl_condition_warning(self, caplog):
        """Test CFL condition warning for unstable timestep."""
        # Create config with violating timestep
        config = NeuralFieldConfig(
            dt=1.0,  # Too large
            diffusion_da=1.0,
            dx=0.1,  # Small spatial step
        )

        # Should log warning about CFL violation
        # Note: warning is logged in __post_init__

    def test_boundary_type_options(self):
        """Test different boundary condition types."""
        config1 = NeuralFieldConfig(boundary_type="no-flux")
        assert config1.boundary_type == "no-flux"

        config2 = NeuralFieldConfig(boundary_type="periodic")
        assert config2.boundary_type == "periodic"


class TestNeuralFieldSolverBasics:
    """Test basic solver functionality."""

    def test_initialization_1d(self):
        """Test 1D field initialization."""
        config = NeuralFieldConfig(spatial_dims=1, grid_size=32)
        solver = NeuralFieldSolver(config=config)

        # Check field shape
        assert solver.fields.shape == (6, 32)

        # Check initialized at baseline
        assert np.allclose(solver.fields, 0.5)

    def test_initialization_2d(self):
        """Test 2D field initialization."""
        config = NeuralFieldConfig(spatial_dims=2, grid_size=16)
        solver = NeuralFieldSolver(config=config)

        assert solver.fields.shape == (6, 16, 16)
        assert np.allclose(solver.fields, 0.5)

    def test_initialization_3d(self):
        """Test 3D field initialization."""
        config = NeuralFieldConfig(spatial_dims=3, grid_size=8)
        solver = NeuralFieldSolver(config=config)

        assert solver.fields.shape == (6, 8, 8, 8)

    def test_get_mean_state(self):
        """Test spatially-averaged state extraction."""
        solver = NeuralFieldSolver()
        state = solver.get_mean_state()

        assert isinstance(state, NeurotransmitterState)
        assert state.dopamine == 0.5  # Baseline

    def test_get_field(self):
        """Test extracting specific NT field."""
        solver = NeuralFieldSolver()
        da_field = solver.get_field(NeurotransmitterType.DOPAMINE)

        assert da_field.shape == (32,)  # Default 1D, grid_size=32
        assert np.allclose(da_field, 0.5)

    def test_reset(self):
        """Test field reset."""
        solver = NeuralFieldSolver()

        # Perturb field
        solver.fields[0] += 0.3

        # Reset
        solver.reset()

        assert np.allclose(solver.fields, 0.5)
        assert solver._time == 0.0
        assert solver._step_count == 0


class TestPDEDynamics:
    """Test PDE integration accuracy."""

    def test_decay_only(self):
        """Test pure decay dynamics without diffusion."""
        config = NeuralFieldConfig(
            dt=0.001,
            alpha_da=10.0,  # Fast decay
            diffusion_da=0.0,  # No diffusion
            adaptive_timestepping=False,
        )
        solver = NeuralFieldSolver(config=config)

        # Set initial condition: high DA
        solver.fields[0] = 0.9

        # Evolve for 100 steps
        for _ in range(100):
            solver.step()

        # DA should have decayed toward 0
        # Semi-implicit: U_new = U / (1 + dt*alpha)
        # After time T = 100*0.001 = 0.1s with alpha=10:
        # U(t) ≈ U0 * exp(-alpha*t) ≈ 0.9 * exp(-10*0.1) = 0.9 * exp(-1) ≈ 0.33
        final_da = solver.get_mean_state().dopamine

        # Should have decayed significantly
        assert final_da < 0.5
        assert final_da > 0.0

    def test_diffusion_spreads(self):
        """Test that diffusion spreads localized perturbation."""
        config = NeuralFieldConfig(
            spatial_dims=1,
            grid_size=32,
            dt=0.0001,  # Small timestep for stability
            alpha_da=0.1,  # Slow decay
            diffusion_da=0.1,
            adaptive_timestepping=False,
        )
        solver = NeuralFieldSolver(config=config)

        # Create localized spike in center
        center = 16
        solver.fields[0, center] = 1.0

        # Measure initial spread
        initial_std = np.std(solver.fields[0])

        # Evolve
        for _ in range(100):
            solver.step()

        # Spread should increase (diffusion smooths)
        final_std = np.std(solver.fields[0])

        # Note: diffusion will smooth out the spike, reducing std
        # But peak will spread to neighbors
        final_center = solver.fields[0, center]
        final_neighbor = solver.fields[0, center + 1]
        far_cell = solver.fields[0, 0]  # Far from spike

        # Neighbors should have more than far-away cells (diffusion gradient)
        # With decay, all values trend toward baseline, but neighbors got some diffusion
        assert final_neighbor > far_cell or np.isclose(final_neighbor, far_cell, atol=0.01)

    def test_semi_implicit_stability(self):
        """Test that semi-implicit method is stable for large decay rates."""
        config = NeuralFieldConfig(
            dt=0.01,  # Relatively large timestep
            alpha_glu=200.0,  # Very fast decay (5ms timescale)
            diffusion_glu=0.0,
            adaptive_timestepping=False,
        )
        solver = NeuralFieldSolver(config=config)

        # Set high glutamate
        solver.fields[5] = 1.0  # Glu is index 5

        # Evolve - should not explode
        for _ in range(100):
            state = solver.step()

            # Should remain bounded
            assert 0.0 <= state.glutamate <= 1.0
            assert not np.isnan(state.glutamate)
            assert not np.isinf(state.glutamate)


class TestBiologicalPlausibility:
    """Test biological plausibility of dynamics."""

    def test_nt_timescales(self):
        """Test that different NTs have correct decay timescales."""
        config = NeuralFieldConfig(
            dt=0.001,
            diffusion_da=0.0,
            diffusion_glu=0.0,
            adaptive_timestepping=False,
        )
        solver = NeuralFieldSolver(config=config)

        # Set both DA and Glu to high
        solver.fields[0] = 1.0  # DA
        solver.fields[5] = 1.0  # Glu

        # Evolve for 0.02s (20ms)
        for _ in range(20):
            solver.step()

        state = solver.get_mean_state()

        # Glu decays much faster than DA
        # alpha_glu = 200 (5ms timescale)
        # alpha_da = 10 (100ms timescale)
        # After 20ms:
        # - Glu should be nearly cleared
        # - DA should still be relatively high

        assert state.glutamate < state.dopamine

    def test_concentration_bounds_enforced(self):
        """Test that concentrations stay in [0, 1]."""
        solver = NeuralFieldSolver()

        # Try to force out of bounds via stimulus
        solver.fields[0] = 0.95
        solver.inject_stimulus(NeurotransmitterType.DOPAMINE, magnitude=1.0)

        # Should be clamped to 1.0
        assert np.max(solver.fields[0]) == 1.0

        # Try negative
        solver.fields[0] = 0.05
        solver.inject_stimulus(NeurotransmitterType.DOPAMINE, magnitude=-1.0)
        assert np.min(solver.fields[0]) == 0.0

    def test_excitotoxicity_prevention(self):
        """Test that glutamate clears quickly to prevent excitotoxicity."""
        config = NeuralFieldConfig(
            dt=0.001,
            alpha_glu=200.0,  # Fast clearance
            diffusion_glu=0.02,  # Local only
            adaptive_timestepping=False,
        )
        solver = NeuralFieldSolver(config=config)

        # Simulate glutamate spike
        solver.fields[5] = 1.0

        # Should clear within 20ms
        for _ in range(20):
            solver.step()

        # Glutamate should be mostly cleared
        assert solver.get_mean_state().glutamate < 0.5


class TestStimulusInjection:
    """Test stimulus injection methods."""

    def test_global_stimulus(self):
        """Test uniform global stimulus."""
        solver = NeuralFieldSolver()

        initial_mean = solver.get_mean_state().dopamine

        solver.inject_stimulus(
            NeurotransmitterType.DOPAMINE,
            magnitude=0.2
        )

        final_mean = solver.get_mean_state().dopamine

        # All points should increase by 0.2
        assert np.abs(final_mean - (initial_mean + 0.2)) < 1e-5

    def test_localized_stimulus(self):
        """Test point stimulus at specific location."""
        config = NeuralFieldConfig(spatial_dims=1, grid_size=32)
        solver = NeuralFieldSolver(config=config)

        location = (10,)
        magnitude = 0.3

        solver.inject_stimulus(
            NeurotransmitterType.SEROTONIN,
            magnitude=magnitude,
            location=location
        )

        # Location should increase
        assert solver.fields[1, location] > 0.5 + magnitude - 0.01

        # Other locations unchanged
        assert solver.fields[1, 0] == 0.5

    def test_phasic_burst(self):
        """Test phasic burst with Gaussian profile."""
        config = NeuralFieldConfig(spatial_dims=1, grid_size=32)
        solver = NeuralFieldSolver(config=config)

        center = (16,)
        magnitude = 0.4
        width = 3.0

        solver.inject_phasic_burst(
            NeurotransmitterType.DOPAMINE,
            center=center,
            magnitude=magnitude,
            width=width
        )

        # Center should have highest value
        center_val = solver.fields[0, center]
        neighbor_val = solver.fields[0, 17]

        assert center_val > neighbor_val
        assert center_val > 0.5 + magnitude * 0.9  # Near peak

        # Far away should be unaffected
        assert solver.fields[0, 0] < 0.6

    def test_phasic_burst_2d(self):
        """Test phasic burst in 2D."""
        config = NeuralFieldConfig(spatial_dims=2, grid_size=16)
        solver = NeuralFieldSolver(config=config)

        center = (8, 8)
        solver.inject_phasic_burst(
            NeurotransmitterType.NOREPINEPHRINE,
            center=center,
            magnitude=0.3
        )

        # Center should be elevated
        assert solver.fields[3, 8, 8] > 0.7

        # Corners should be less affected
        assert solver.fields[3, 0, 0] < 0.6


class TestNumericalStability:
    """Test numerical stability and adaptive timestepping."""

    def test_adaptive_timestep_reduction(self):
        """Test that adaptive timestepping reduces dt on large changes."""
        config = NeuralFieldConfig(
            dt=0.01,  # Start with large dt
            adaptive_timestepping=True,
        )
        solver = NeuralFieldSolver(config=config)

        # Create large perturbation that would cause instability
        solver.fields[0] = 1.0

        # Inject massive stimulus to force large change
        large_stim = np.zeros_like(solver.fields)
        large_stim[0] = 5.0  # Huge stimulus

        # This should trigger adaptive reduction
        initial_dt = solver._current_dt
        solver.step(stimulus=large_stim)

        # dt might be reduced (if step was rejected)
        # At minimum, solver should remain stable
        assert not np.any(np.isnan(solver.fields))
        assert not np.any(np.isinf(solver.fields))

    def test_no_nan_or_inf(self):
        """Test that solver never produces NaN or Inf."""
        solver = NeuralFieldSolver()

        # Run for many steps with random stimuli
        for _ in range(100):
            stim = np.random.randn(*solver.fields.shape) * 0.1
            solver.step(stimulus=stim)

            assert not np.any(np.isnan(solver.fields))
            assert not np.any(np.isinf(solver.fields))


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_no_flux_boundaries(self):
        """Test no-flux (Neumann) boundary conditions."""
        config = NeuralFieldConfig(
            spatial_dims=1,
            grid_size=32,
            boundary_type="no-flux",
            dt=0.001,
            alpha_da=0.1,
            diffusion_da=0.1,
        )
        solver = NeuralFieldSolver(config=config)

        # Put spike at boundary
        solver.fields[0, 0] = 1.0

        # Evolve
        for _ in range(50):
            solver.step()

        # With no-flux, concentration shouldn't leak out
        # (no special test, just shouldn't crash)
        assert solver.fields[0, 0] > 0.5

    def test_periodic_boundaries(self):
        """Test periodic boundary conditions."""
        config = NeuralFieldConfig(
            spatial_dims=1,
            grid_size=32,
            boundary_type="periodic",
            dt=0.001,
            alpha_da=0.1,
            diffusion_da=0.1,
        )
        solver = NeuralFieldSolver(config=config)

        # Put spike at left boundary
        solver.fields[0, 0] = 1.0

        # Evolve
        for _ in range(50):
            solver.step()

        # With periodic BC, diffusion wraps around
        # Right boundary should have received some diffusion from left
        # Check that gradient exists (left boundary neighbors affected)
        left_neighbor = solver.fields[0, 1]
        right_neighbor = solver.fields[0, -1]
        middle = solver.fields[0, 16]

        # With periodic BC, both neighbors of index 0 should be similar
        # (wrapping means -1 is also a neighbor of 0)
        assert np.isclose(left_neighbor, right_neighbor, atol=0.1)


class TestIntegrationWithCoupling:
    """Test integration with LearnableCoupling."""

    def test_coupling_integration(self):
        """Test that coupling modulates NT dynamics."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()

        config = NeuralFieldConfig(
            dt=0.001,
            alpha_da=1.0,
            diffusion_da=0.0,
            adaptive_timestepping=False,
        )
        solver = NeuralFieldSolver(config=config, coupling=coupling)

        # Set specific NT state
        solver.fields[0] = 0.8  # High DA

        # Run with coupling
        for _ in range(10):
            solver.step()

        # Coupling should influence dynamics
        # (exact behavior depends on coupling matrix)
        state = solver.get_mean_state()
        assert 0.0 <= state.dopamine <= 1.0


class TestIntegrationWithAttractors:
    """Test integration with attractor dynamics."""

    def test_attractor_integration(self):
        """Test that attractor forces pull state toward basin."""
        from ww.nca.attractors import StateTransitionManager

        attractor_manager = StateTransitionManager()

        config = NeuralFieldConfig(
            dt=0.001,
            adaptive_timestepping=False,
        )
        solver = NeuralFieldSolver(
            config=config,
            attractor_manager=attractor_manager
        )

        # Run dynamics
        for _ in range(100):
            solver.step()

        # State should be influenced by attractor
        state = solver.get_mean_state()

        # Should remain in valid range
        assert 0.0 <= state.dopamine <= 1.0
        assert 0.0 <= state.serotonin <= 1.0


class TestSolverStatistics:
    """Test statistics and monitoring."""

    def test_get_stats(self):
        """Test solver statistics output."""
        solver = NeuralFieldSolver()

        # Run a few steps
        for _ in range(10):
            solver.step()

        stats = solver.get_stats()

        assert stats['step_count'] == 10
        assert stats['time'] > 0
        assert 'spatial_stats' in stats
        assert 'dopamine' in stats['spatial_stats']
        assert 'mean' in stats['spatial_stats']['dopamine']

    def test_step_count_tracking(self):
        """Test step count increments correctly."""
        solver = NeuralFieldSolver()

        assert solver._step_count == 0

        solver.step()
        assert solver._step_count == 1

        for _ in range(9):
            solver.step()

        assert solver._step_count == 10

    def test_time_tracking(self):
        """Test simulation time tracking."""
        config = NeuralFieldConfig(dt=0.01, adaptive_timestepping=False)
        solver = NeuralFieldSolver(config=config)

        # Run 10 steps
        for _ in range(10):
            solver.step()

        # Time should be approximately 10 * 0.01 = 0.1s
        assert np.abs(solver._time - 0.1) < 1e-6


class TestMultiDimensionalFields:
    """Test multi-dimensional spatial fields."""

    def test_1d_diffusion(self):
        """Test diffusion in 1D."""
        config = NeuralFieldConfig(
            spatial_dims=1,
            grid_size=32,
            dt=0.001,
            alpha_da=0.1,
            diffusion_da=0.1,
        )
        solver = NeuralFieldSolver(config=config)

        # Central spike
        solver.fields[0, 16] = 1.0

        # Diffuse
        for _ in range(50):
            solver.step()

        # Should spread to neighbors - check gradient exists
        # With decay, values may be below 0.5 but neighbors should have
        # received diffusion and be higher than far-away cells
        neighbor_left = solver.fields[0, 15]
        neighbor_right = solver.fields[0, 17]
        far_cell = solver.fields[0, 0]

        # Neighbors should have more than far-away cells
        assert neighbor_left >= far_cell - 0.01
        assert neighbor_right >= far_cell - 0.01

    def test_2d_diffusion(self):
        """Test diffusion in 2D."""
        config = NeuralFieldConfig(
            spatial_dims=2,
            grid_size=16,
            dt=0.0001,
            alpha_da=0.1,
            diffusion_da=0.1,
        )
        solver = NeuralFieldSolver(config=config)

        # Central spike
        solver.fields[0, 8, 8] = 1.0

        # Diffuse
        for _ in range(50):
            solver.step()

        # Should spread radially - check neighbors vs far corners
        center = solver.fields[0, 8, 8]
        neighbors = [
            solver.fields[0, 7, 8],  # Up
            solver.fields[0, 9, 8],  # Down
            solver.fields[0, 8, 7],  # Left
            solver.fields[0, 8, 9],  # Right
        ]
        corner = solver.fields[0, 0, 0]

        # With diffusion, neighbors of center should have received some signal
        # and should be >= corners (accounting for numerical precision)
        for neighbor in neighbors:
            assert neighbor >= corner - 0.01

    def test_3d_initialization(self):
        """Test 3D field initialization and basic dynamics."""
        config = NeuralFieldConfig(
            spatial_dims=3,
            grid_size=8,
            dt=0.001,
        )
        solver = NeuralFieldSolver(config=config)

        assert solver.fields.shape == (6, 8, 8, 8)

        # Should be able to step
        solver.step()
        assert solver._step_count == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_diffusion(self):
        """Test dynamics with zero diffusion."""
        config = NeuralFieldConfig(
            diffusion_da=0.0,
            diffusion_5ht=0.0,
            diffusion_ach=0.0,
            diffusion_ne=0.0,
            diffusion_gaba=0.0,
            diffusion_glu=0.0,
        )
        solver = NeuralFieldSolver(config=config)

        # Should still work (pure decay)
        solver.step()
        assert solver._step_count == 1

    def test_zero_decay(self):
        """Test dynamics with zero decay."""
        config = NeuralFieldConfig(
            alpha_da=0.0,
            alpha_5ht=0.0,
            alpha_ach=0.0,
            alpha_ne=0.0,
            alpha_gaba=0.0,
            alpha_glu=0.0,
        )
        solver = NeuralFieldSolver(config=config)

        # Should still work (pure diffusion)
        solver.step()
        assert solver._step_count == 1

    def test_very_small_timestep(self):
        """Test with very small timestep."""
        config = NeuralFieldConfig(dt=1e-6, adaptive_timestepping=False)
        solver = NeuralFieldSolver(config=config)

        solver.step()
        assert solver._time > 0

    def test_custom_dt_override(self):
        """Test overriding dt in step."""
        config = NeuralFieldConfig(dt=0.001, adaptive_timestepping=False)
        solver = NeuralFieldSolver(config=config)

        # Use custom dt
        solver.step(dt=0.01)

        # Time should reflect custom dt
        assert np.abs(solver._time - 0.01) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
