"""
Tests for transmission delay system.

Validates:
1. Circular delay buffer operations
2. Distance matrix and delay calculations
3. NT-specific synaptic delays
4. Region-to-region transmission delays
5. Neural field integration
6. Delay differential operators
"""

import numpy as np
import pytest

from ww.nca.delays import (
    CircularDelayBuffer,
    DelayConfig,
    DelayDifferentialOperator,
    DelayState,
    DistanceMatrix,
    FiberType,
    TransmissionDelaySystem,
    compute_axonal_delay,
    create_delay_system,
)


class TestDelayConfig:
    """Test delay configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DelayConfig()
        assert config.dt_ms == 1.0
        assert config.max_delay_ms == 100.0
        assert config.default_velocity_m_s == 5.0

    def test_buffer_size_calculation(self):
        """Test buffer size is correctly computed."""
        config = DelayConfig(max_delay_ms=100.0, dt_ms=1.0)
        assert config.buffer_size == 101  # 100/1 + 1

        config2 = DelayConfig(max_delay_ms=50.0, dt_ms=2.0)
        assert config2.buffer_size == 26  # 50/2 + 1

    def test_synaptic_delays(self):
        """Test NT-specific synaptic delays."""
        config = DelayConfig()

        # Fast synapses (Glu, GABA)
        assert config.get_synaptic_delay(5) < 5  # Glu
        assert config.get_synaptic_delay(4) < 5  # GABA

        # Volume transmission (DA, 5HT, NE) - slower
        assert config.get_synaptic_delay(0) > 10  # DA
        assert config.get_synaptic_delay(1) > 10  # 5HT
        assert config.get_synaptic_delay(3) > 10  # NE

    def test_biological_delay_ordering(self):
        """Test delays follow biological ordering."""
        config = DelayConfig()

        # Glutamate should be fastest
        glu_delay = config.get_synaptic_delay(5)
        gaba_delay = config.get_synaptic_delay(4)
        ach_delay = config.get_synaptic_delay(2)
        da_delay = config.get_synaptic_delay(0)

        assert glu_delay < gaba_delay < ach_delay < da_delay


class TestCircularDelayBuffer:
    """Test circular buffer for signal delay."""

    def test_initialization(self):
        """Test buffer initializes to zeros."""
        buffer = CircularDelayBuffer(size=10, shape=(4,))
        assert buffer.size == 10
        assert buffer.shape == (4,)
        assert np.allclose(buffer.get_current(), np.zeros(4))

    def test_push_and_retrieve(self):
        """Test pushing and retrieving values."""
        buffer = CircularDelayBuffer(size=5, shape=(2,))

        # Push a sequence
        for i in range(5):
            buffer.push(np.array([i, i + 10], dtype=np.float32))

        # Current (most recent) should be [4, 14]
        assert np.allclose(buffer.get_current(), [4, 14])

        # Oldest should be [0, 10]
        assert np.allclose(buffer.get_oldest(), [0, 10])

    def test_delayed_retrieval(self):
        """Test getting values with specific delays."""
        buffer = CircularDelayBuffer(size=10, shape=())

        # Push values 0-9
        for i in range(10):
            buffer.push(np.array(float(i)))

        # Current (0 delay) = 9
        assert buffer.get_delayed(0) == pytest.approx(9.0)

        # 3 steps ago = 6
        assert buffer.get_delayed(3) == pytest.approx(6.0)

        # 9 steps ago = 0 (oldest)
        assert buffer.get_delayed(9) == pytest.approx(0.0)

    def test_interpolated_delay(self):
        """Test fractional delay interpolation."""
        buffer = CircularDelayBuffer(size=10, shape=())

        for i in range(10):
            buffer.push(np.array(float(i)))

        # 1.5 steps ago should be between 7 and 8 = 7.5
        assert buffer.interpolate_delay(1.5) == pytest.approx(7.5)

        # 2.25 steps ago = 0.75*7 + 0.25*6 = 6.75
        assert buffer.interpolate_delay(2.25) == pytest.approx(6.75)

    def test_circular_wraparound(self):
        """Test buffer correctly wraps around."""
        buffer = CircularDelayBuffer(size=5, shape=())

        # Push 20 values (4 full cycles)
        for i in range(20):
            buffer.push(np.array(float(i)))

        # Should only have last 5: 15, 16, 17, 18, 19
        assert buffer.get_current() == pytest.approx(19.0)
        assert buffer.get_oldest() == pytest.approx(15.0)

    def test_clear(self):
        """Test buffer clearing."""
        buffer = CircularDelayBuffer(size=5, shape=(3,))

        buffer.push(np.array([1.0, 2.0, 3.0]))
        buffer.clear()

        assert np.allclose(buffer.get_current(), np.zeros(3))

    def test_multidimensional_shape(self):
        """Test buffer with multi-dimensional data."""
        buffer = CircularDelayBuffer(size=5, shape=(2, 3))

        data = np.ones((2, 3)) * 5
        buffer.push(data)

        assert buffer.get_current().shape == (2, 3)
        assert np.allclose(buffer.get_current(), data)


class TestDistanceMatrix:
    """Test distance matrix for region delays."""

    def test_initialization(self):
        """Test matrix initialization."""
        dm = DistanceMatrix(n_regions=4, default_distance=20.0)

        assert dm.n_regions == 4
        # Diagonal should be zero
        assert dm.get_distance(0, 0) == 0.0
        # Off-diagonal should be default
        assert dm.get_distance(0, 1) == 20.0

    def test_symmetric(self):
        """Test distances are symmetric."""
        dm = DistanceMatrix(n_regions=3)

        dm.set_distance(0, 2, 50.0)
        assert dm.get_distance(0, 2) == 50.0
        assert dm.get_distance(2, 0) == 50.0

    def test_delay_calculation(self):
        """Test delay from distance and velocity."""
        dm = DistanceMatrix(n_regions=2)
        dm.set_distance(0, 1, 10.0)  # 10 mm

        # At 10 m/s = 10 mm/ms
        # 10 mm / (10 mm/ms) = 1 ms
        delay = dm.get_delay_ms(0, 1, velocity_m_s=10.0)
        assert delay == pytest.approx(1.0)

        # At 5 m/s = 5 mm/ms
        # 10 mm / (5 mm/ms) = 2 ms
        delay2 = dm.get_delay_ms(0, 1, velocity_m_s=5.0)
        assert delay2 == pytest.approx(2.0)

    def test_from_coordinates(self):
        """Test creating from 3D coordinates."""
        coords = np.array([
            [0, 0, 0],
            [10, 0, 0],  # 10 mm away
            [0, 10, 0],  # 10 mm away
        ], dtype=np.float32)

        dm = DistanceMatrix.from_coordinates(coords)

        assert dm.get_distance(0, 1) == pytest.approx(10.0)
        assert dm.get_distance(0, 2) == pytest.approx(10.0)
        assert dm.get_distance(1, 2) == pytest.approx(np.sqrt(200))  # ~14.14


class TestTransmissionDelaySystem:
    """Test complete transmission delay system."""

    def test_initialization(self):
        """Test system initialization."""
        system = TransmissionDelaySystem(
            n_regions=2,
            grid_shape=(8,)
        )

        assert system.n_regions == 2
        assert system.grid_shape == (8,)

    def test_push_and_get_nt(self):
        """Test pushing and retrieving NT fields."""
        system = TransmissionDelaySystem(grid_shape=(4,))

        # Create test field
        fields = np.zeros((6, 4), dtype=np.float32)
        fields[0] = 1.0  # DA = 1

        # Push multiple times to fill buffer
        for _ in range(30):  # DA delay is 20ms
            system.push_nt_state(fields)

        # Get delayed DA (should be close to what we pushed)
        delayed_da = system.get_delayed_nt(0)  # DA
        assert delayed_da.shape == (4,)
        assert np.allclose(delayed_da, 1.0)

    def test_different_nt_delays(self):
        """Test different NTs have different delays."""
        system = TransmissionDelaySystem(grid_shape=(4,))

        # Fill buffer with time-varying signal
        for t in range(50):
            fields = np.ones((6, 4)) * t
            system.push_nt_state(fields)

        # Get delays for different NTs
        glu_delayed = system.get_delayed_nt(5)  # Fast
        da_delayed = system.get_delayed_nt(0)   # Slow

        # Glu (fast) should be closer to current (t=49)
        # DA (slow) should be further back
        assert np.mean(glu_delayed) > np.mean(da_delayed)

    def test_region_to_region_delay(self):
        """Test inter-region transmission."""
        system = TransmissionDelaySystem(
            n_regions=3,
            grid_shape=(4,)
        )

        # Set custom distance
        system.set_region_distance(0, 2, 50.0)  # 50 mm

        # Push fields for region 0 → region 2
        fields = np.ones((6, 4)) * 10.0
        for _ in range(30):
            system.push_region_state(0, 2, fields)

        # Get delayed input
        delayed = system.get_delayed_region_input(0, 2)
        assert delayed.shape == (6, 4)

    def test_velocity_setting(self):
        """Test setting custom velocities."""
        system = TransmissionDelaySystem(n_regions=2)

        # Default velocity
        delay1 = system.get_delay_ms(0, 1)

        # Increase velocity
        system.set_velocity(0, 1, 20.0)  # 20 m/s
        delay2 = system.get_delay_ms(0, 1)

        # Faster velocity = shorter delay
        assert delay2 < delay1

    def test_delayed_coupling(self):
        """Test coupling with delays."""
        system = TransmissionDelaySystem(grid_shape=(4,))

        # Fill buffer
        for t in range(50):
            fields = np.ones((6, 4)) * 0.5
            system.push_nt_state(fields)

        # Create coupling matrix
        coupling = np.eye(6) * 0.1

        # Compute delayed coupling
        current = np.ones((6, 4)) * 0.5
        result = system.compute_delayed_coupling(current, coupling)

        assert result.shape == (6, 4)

    def test_reset(self):
        """Test system reset."""
        system = TransmissionDelaySystem(grid_shape=(4,))

        # Add some data
        fields = np.ones((6, 4))
        for _ in range(10):
            system.push_nt_state(fields)

        system.reset()

        assert system._step_count == 0

    def test_stats(self):
        """Test statistics retrieval."""
        system = TransmissionDelaySystem(n_regions=2, grid_shape=(8,))

        stats = system.get_stats()

        assert "buffer_size" in stats
        assert "n_regions" in stats
        assert "synaptic_delays" in stats
        assert "da" in stats["synaptic_delays"]


class TestDelayDifferentialOperator:
    """Test delay differential equation operator."""

    def test_initialization(self):
        """Test operator initialization."""
        system = TransmissionDelaySystem(grid_shape=(4,))
        operator = DelayDifferentialOperator(system)

        assert operator.delay_system is system
        assert operator.delay_weights.shape == (6, 6)

    def test_delay_term_computation(self):
        """Test delay term computation."""
        system = TransmissionDelaySystem(grid_shape=(4,))
        operator = DelayDifferentialOperator(system)

        # Fill buffer
        for _ in range(50):
            fields = np.ones((6, 4)) * 0.5
            system.push_nt_state(fields)

        # Compute delay term
        current = np.ones((6, 4)) * 0.5
        delay_term = operator.compute_delay_term(current)

        assert delay_term.shape == (6, 4)

    def test_step_advances_buffer(self):
        """Test that step advances delay buffer."""
        system = TransmissionDelaySystem(grid_shape=(4,))
        operator = DelayDifferentialOperator(system)

        initial_count = system._step_count

        fields = np.ones((6, 4))
        operator.step(fields)

        assert system._step_count == initial_count + 1


class TestNeuralFieldIntegration:
    """Test integration with neural field solver."""

    def test_solver_with_delays(self):
        """Test neural field solver accepts delay system."""
        from ww.nca import NeuralFieldSolver, TransmissionDelaySystem

        delay_system = TransmissionDelaySystem(grid_shape=(16,))
        solver = NeuralFieldSolver(delay_system=delay_system)

        assert solver.delay_system is delay_system

    def test_delays_affect_dynamics(self):
        """Test delays influence field dynamics."""
        from ww.nca import NeuralFieldSolver, NeuralFieldConfig, TransmissionDelaySystem

        # Without delays
        solver_no_delay = NeuralFieldSolver()

        # With delays - match default grid size (32)
        delay_system = TransmissionDelaySystem(grid_shape=(32,))
        solver_delay = NeuralFieldSolver(delay_system=delay_system)

        # Run both for several steps
        for _ in range(20):
            state_no_delay = solver_no_delay.step()
            state_delay = solver_delay.step()

        # Both should be stable (no NaN/Inf)
        assert not np.isnan(state_no_delay.dopamine)
        assert not np.isnan(state_delay.dopamine)

    def test_get_transmission_delay(self):
        """Test transmission delay accessor."""
        from ww.nca import NeuralFieldSolver, TransmissionDelaySystem

        delay_system = TransmissionDelaySystem()
        solver = NeuralFieldSolver(delay_system=delay_system)

        # Get DA delay
        da_delay = solver.get_transmission_delay(0)
        assert da_delay > 0

        # Glu should be faster than DA
        glu_delay = solver.get_transmission_delay(5)
        assert glu_delay < da_delay

    def test_no_delay_returns_zero(self):
        """Test solver without delays returns zero delay."""
        from ww.nca import NeuralFieldSolver

        solver = NeuralFieldSolver()  # No delay system
        assert solver.get_transmission_delay(0) == 0.0


class TestFiberTypes:
    """Test axon fiber type delays."""

    def test_fiber_velocities(self):
        """Test different fiber types have different velocities."""
        distance = 10.0  # 10 mm

        delay_c = compute_axonal_delay(distance, FiberType.C_UNMYELINATED)
        delay_a_delta = compute_axonal_delay(distance, FiberType.A_DELTA)
        delay_a_alpha = compute_axonal_delay(distance, FiberType.A_ALPHA)

        # C fibers (slowest) > A-delta > A-alpha (fastest)
        assert delay_c > delay_a_delta > delay_a_alpha

    def test_cortical_fibers(self):
        """Test cortical fiber delays."""
        distance = 50.0  # 50 mm

        local = compute_axonal_delay(distance, FiberType.CORTICAL_LOCAL)
        long_range = compute_axonal_delay(distance, FiberType.CORTICAL_LONG)

        # Local (slower) > long-range (faster, myelinated)
        assert local > long_range

    def test_delay_scales_with_distance(self):
        """Test delay increases with distance."""
        delay_10mm = compute_axonal_delay(10.0, FiberType.CORTICAL_LONG)
        delay_50mm = compute_axonal_delay(50.0, FiberType.CORTICAL_LONG)

        assert delay_50mm == pytest.approx(delay_10mm * 5)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_delay_system(self):
        """Test factory function."""
        system = create_delay_system(
            n_regions=3,
            grid_size=8,
            max_delay_ms=50.0
        )

        assert system.n_regions == 3
        assert system.grid_shape == (8,)
        assert system.config.max_delay_ms == 50.0

    def test_compute_axonal_delay(self):
        """Test axonal delay computation."""
        # 10 mm at 5 m/s = 2 ms
        delay = compute_axonal_delay(10.0, FiberType.CORTICAL_LONG)
        assert delay == pytest.approx(2.0)


class TestVelocityPlasticity:
    """Test activity-dependent velocity changes."""

    def test_plasticity_disabled_by_default(self):
        """Test plasticity is off by default."""
        config = DelayConfig()
        assert not config.delay_plasticity

    def test_velocity_update_when_enabled(self):
        """Test velocity changes with plasticity enabled."""
        config = DelayConfig(delay_plasticity=True, plasticity_rate=0.1)
        system = TransmissionDelaySystem(config=config, n_regions=2)

        initial_velocity = system._velocities[0, 1]

        # High activity should increase velocity
        for _ in range(100):
            system.update_velocity_plasticity(0, 1, activity=0.9)

        assert system._velocities[0, 1] > initial_velocity

    def test_velocity_bounded(self):
        """Test velocity doesn't exceed biological maximum."""
        config = DelayConfig(delay_plasticity=True, plasticity_rate=1.0)
        system = TransmissionDelaySystem(config=config, n_regions=2)

        # Many high-activity updates
        for _ in range(1000):
            system.update_velocity_plasticity(0, 1, activity=1.0)

        # Should not exceed ~100 m/s (biological max)
        assert system._velocities[0, 1] <= 100.0


class TestDelayState:
    """Test delay state tracking."""

    def test_state_attributes(self):
        """Test state has expected attributes."""
        state = DelayState()

        assert hasattr(state, "total_buffered_ms")
        assert hasattr(state, "active_delays")
        assert hasattr(state, "mean_delay_ms")
        assert hasattr(state, "max_active_delay_ms")

    def test_get_state_updates(self):
        """Test state updates with usage."""
        system = TransmissionDelaySystem(grid_shape=(4,))

        # Initially no time buffered
        state1 = system.get_state()
        assert state1.total_buffered_ms == 0.0

        # After stepping
        fields = np.ones((6, 4))
        for _ in range(10):
            system.step(fields)

        state2 = system.get_state()
        assert state2.total_buffered_ms > 0
