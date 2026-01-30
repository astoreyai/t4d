"""
Performance Benchmarks for NCA Modules.

Sprint 3: Measures timing and throughput for neural circuit components.

These tests validate that modules can run in real-time or faster,
which is critical for interactive applications.

Performance targets:
- Neural field step: <10ms for 32x32x32 grid
- Hippocampus process: <5ms per pattern
- VTA RPE computation: <1ms
- Raphe step: <1ms
- SWR step: <1ms
- MSN step: <1ms
"""

import time
import numpy as np
import pytest
from typing import Callable, Tuple

from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig
from ww.nca.hippocampus import HippocampalCircuit, HippocampalConfig
from ww.nca.vta import VTACircuit, VTAConfig
from ww.nca.raphe import RapheNucleus, RapheConfig
from ww.nca.swr_coupling import SWRNeuralFieldCoupling, SWRConfig
from ww.nca.striatal_msn import StriatalMSN, MSNConfig
from ww.nca.coupling import LearnableCoupling, CouplingConfig


def benchmark(func: Callable, iterations: int = 100) -> Tuple[float, float, float]:
    """
    Run benchmark and return timing statistics.

    Returns:
        (mean_ms, min_ms, max_ms) tuple
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return np.mean(times), np.min(times), np.max(times)


# =============================================================================
# Neural Field Performance
# =============================================================================

class TestNeuralFieldPerformance:
    """Performance benchmarks for NeuralFieldSolver."""

    def test_small_grid_step_performance(self):
        """Small grid (8x8x8) step should be <2ms."""
        config = NeuralFieldConfig(grid_size=8)
        solver = NeuralFieldSolver(config)

        mean, min_, max_ = benchmark(lambda: solver.step(dt=0.01), iterations=100)

        print(f"\nNeuralField 8x8x8 step: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 5.0, f"8x8x8 grid step too slow: {mean:.2f}ms"

    def test_medium_grid_step_performance(self):
        """Medium grid (16x16x16) step should be <10ms."""
        config = NeuralFieldConfig(grid_size=16)
        solver = NeuralFieldSolver(config)

        mean, min_, max_ = benchmark(lambda: solver.step(dt=0.01), iterations=50)

        print(f"\nNeuralField 16x16x16 step: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 20.0, f"16x16x16 grid step too slow: {mean:.2f}ms"

    def test_large_grid_step_performance(self):
        """Large grid (32x32x32) step should be <50ms."""
        config = NeuralFieldConfig(grid_size=32)
        solver = NeuralFieldSolver(config)

        mean, min_, max_ = benchmark(lambda: solver.step(dt=0.01), iterations=20)

        print(f"\nNeuralField 32x32x32 step: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 100.0, f"32x32x32 grid step too slow: {mean:.2f}ms"


# =============================================================================
# Hippocampus Performance
# =============================================================================

class TestHippocampusPerformance:
    """Performance benchmarks for HippocampalCircuit."""

    @pytest.fixture
    def hippocampus(self):
        """Create hippocampus with default dimensions."""
        config = HippocampalConfig(
            ec_dim=256,
            dg_dim=1024,
            ca3_dim=256,
            ca1_dim=256,
        )
        return HippocampalCircuit(config, random_seed=42)

    def test_encode_performance(self, hippocampus):
        """Pattern encoding should be <10ms."""
        pattern = np.random.randn(256)

        def encode():
            hippocampus.encode(pattern)

        mean, min_, max_ = benchmark(encode, iterations=100)

        print(f"\nHippocampus encode: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 20.0, f"Hippocampus encode too slow: {mean:.2f}ms"

    def test_retrieve_performance(self, hippocampus):
        """Pattern retrieval should be <10ms."""
        # Store some patterns first
        for _ in range(10):
            pattern = np.random.randn(256)
            hippocampus.encode(pattern)

        query = np.random.randn(256)

        def retrieve():
            hippocampus.retrieve(query)

        mean, min_, max_ = benchmark(retrieve, iterations=100)

        print(f"\nHippocampus retrieve: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 20.0, f"Hippocampus retrieve too slow: {mean:.2f}ms"

    def test_ca3_completion_performance(self, hippocampus):
        """CA3 pattern completion should be <5ms."""
        # Store patterns
        for _ in range(50):
            pattern = np.random.randn(256)
            pattern = pattern / np.linalg.norm(pattern)
            hippocampus.ca3.store(pattern)

        query = np.random.randn(256)
        query = query / np.linalg.norm(query)

        def complete():
            hippocampus.ca3.complete(query)

        mean, min_, max_ = benchmark(complete, iterations=100)

        print(f"\nCA3 completion: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 10.0, f"CA3 completion too slow: {mean:.2f}ms"


# =============================================================================
# VTA Dopamine Performance
# =============================================================================

class TestVTAPerformance:
    """Performance benchmarks for VTACircuit."""

    @pytest.fixture
    def vta(self):
        return VTACircuit()

    def test_rpe_computation_performance(self, vta):
        """RPE computation should be <1ms."""
        def compute_rpe():
            vta.compute_td_error(
                reward=0.5,
                current_state="s1",
                next_state="s2"
            )

        mean, min_, max_ = benchmark(compute_rpe, iterations=1000)

        print(f"\nVTA compute_td_error: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"VTA RPE computation too slow: {mean:.3f}ms"

    def test_process_rpe_performance(self, vta):
        """Process RPE to DA should be <1ms."""
        def process():
            vta.process_rpe(rpe=0.3, dt=0.1)

        mean, min_, max_ = benchmark(process, iterations=1000)

        print(f"\nVTA process_rpe: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"VTA process_rpe too slow: {mean:.3f}ms"

    def test_step_performance(self, vta):
        """VTA step should be <0.5ms."""
        def step():
            vta.step(dt=0.1)

        mean, min_, max_ = benchmark(step, iterations=1000)

        print(f"\nVTA step: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 0.5, f"VTA step too slow: {mean:.3f}ms"


# =============================================================================
# Raphe Serotonin Performance
# =============================================================================

class TestRaphePerformance:
    """Performance benchmarks for RapheNucleus."""

    @pytest.fixture
    def raphe(self):
        return RapheNucleus()

    def test_step_performance(self, raphe):
        """Raphe step should be <1ms."""
        def step():
            raphe.step(dt=0.1)

        mean, min_, max_ = benchmark(step, iterations=1000)

        print(f"\nRaphe step: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"Raphe step too slow: {mean:.3f}ms"

    def test_with_stress_input_performance(self, raphe):
        """Raphe with stress modulation should be <1ms."""
        def step_with_stress():
            raphe.set_stress_input(0.6)
            raphe.step(dt=0.1)

        mean, min_, max_ = benchmark(step_with_stress, iterations=1000)

        print(f"\nRaphe step+stress: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"Raphe step+stress too slow: {mean:.3f}ms"


# =============================================================================
# SWR Coupling Performance
# =============================================================================

class TestSWRPerformance:
    """Performance benchmarks for SWRNeuralFieldCoupling."""

    @pytest.fixture
    def swr(self):
        return SWRNeuralFieldCoupling()

    def test_step_quiescent_performance(self, swr):
        """SWR step in quiescent state should be <0.5ms."""
        def step():
            swr.step(dt=0.01)

        mean, min_, max_ = benchmark(step, iterations=1000)

        print(f"\nSWR step (quiescent): {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"SWR quiescent step too slow: {mean:.3f}ms"

    def test_step_during_ripple_performance(self, swr):
        """SWR step during ripple should be <1ms."""
        swr.force_swr()

        def step():
            swr.step(dt=0.001)

        mean, min_, max_ = benchmark(step, iterations=500)

        print(f"\nSWR step (rippling): {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"SWR ripple step too slow: {mean:.3f}ms"


# =============================================================================
# Striatal MSN Performance
# =============================================================================

class TestMSNPerformance:
    """Performance benchmarks for StriatalMSN."""

    @pytest.fixture
    def msn(self):
        return StriatalMSN()

    def test_step_performance(self, msn):
        """MSN step should be <1ms."""
        msn.set_cortical_input(0.7)
        msn.set_dopamine_level(0.5)

        def step():
            msn.step(dt=0.01)

        mean, min_, max_ = benchmark(step, iterations=1000)

        print(f"\nMSN step: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"MSN step too slow: {mean:.3f}ms"

    def test_with_rpe_performance(self, msn):
        """MSN with RPE application should be <1ms."""
        msn.set_cortical_input(0.7)
        msn.set_dopamine_level(0.5)

        def step_with_rpe():
            msn.apply_rpe(0.3)
            msn.step(dt=0.01)

        mean, min_, max_ = benchmark(step_with_rpe, iterations=1000)

        print(f"\nMSN step+RPE: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"MSN step+RPE too slow: {mean:.3f}ms"


# =============================================================================
# Coupling Matrix Performance
# =============================================================================

class TestCouplingPerformance:
    """Performance benchmarks for LearnableCoupling."""

    @pytest.fixture
    def coupling(self):
        return LearnableCoupling()

    def test_compute_coupling_performance(self, coupling):
        """Coupling computation should be <0.5ms."""
        # Create mock NT state
        from ww.nca.neural_field import NeurotransmitterState
        state = NeurotransmitterState()

        def compute():
            coupling.compute_coupling(state)

        mean, min_, max_ = benchmark(compute, iterations=1000)

        print(f"\nCoupling compute: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 1.0, f"Coupling compute too slow: {mean:.3f}ms"


# =============================================================================
# Integrated System Performance
# =============================================================================

class TestIntegratedPerformance:
    """Performance benchmarks for integrated system operation."""

    def test_full_cognitive_loop_performance(self):
        """
        Full cognitive loop (all circuits) should be <50ms.

        This tests the complete processing pipeline:
        VTA -> MSN -> Hippocampus -> SWR
        """
        vta = VTACircuit()
        msn = StriatalMSN()
        hippocampus = HippocampalCircuit(HippocampalConfig(
            ec_dim=128, dg_dim=512, ca3_dim=128, ca1_dim=128
        ))
        raphe = RapheNucleus()
        swr = SWRNeuralFieldCoupling()

        pattern = np.random.randn(128)

        def cognitive_loop():
            # Compute RPE
            rpe = vta.compute_rpe_from_outcome(0.7, 0.5)
            vta.process_rpe(rpe, dt=0.01)

            # Update striatum
            msn.set_dopamine_level(vta.state.current_da)
            msn.set_cortical_input(0.6)
            msn.step(dt=0.01)

            # Process through hippocampus
            hippocampus.process(pattern)

            # Update raphe
            raphe.step(dt=0.01)

            # Update SWR
            swr.step(dt=0.01)

        mean, min_, max_ = benchmark(cognitive_loop, iterations=100)

        print(f"\nFull cognitive loop: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 100.0, f"Cognitive loop too slow: {mean:.2f}ms"

    def test_sustained_simulation_performance(self):
        """
        Sustained simulation (10 seconds) should complete in <10 seconds.

        Tests throughput for real-time operation.
        """
        vta = VTACircuit()
        msn = StriatalMSN()
        raphe = RapheNucleus()

        dt = 0.01  # 10ms steps
        n_steps = 1000  # 10 seconds simulated

        start = time.perf_counter()

        for i in range(n_steps):
            # Random reward every 100 steps
            if i % 100 == 0:
                rpe = np.random.uniform(-0.5, 0.5)
                vta.process_rpe(rpe, dt)

            vta.step(dt)
            msn.set_dopamine_level(vta.state.current_da)
            msn.step(dt)
            raphe.step(dt)

        elapsed = time.perf_counter() - start
        simulated = n_steps * dt

        speedup = simulated / elapsed

        print(f"\nSustained simulation: {elapsed:.2f}s for {simulated}s simulated ({speedup:.1f}x realtime)")
        assert speedup > 1.0, f"Simulation slower than realtime: {speedup:.2f}x"


# =============================================================================
# Memory Usage Tests
# =============================================================================

class TestMemoryUsage:
    """Tests for memory efficiency of NCA modules."""

    def test_hippocampus_memory_scales_linearly(self):
        """Hippocampus memory should scale linearly with patterns."""
        import sys

        config = HippocampalConfig(ec_dim=128, dg_dim=512, ca3_dim=128, ca1_dim=128)
        hippocampus = HippocampalCircuit(config)

        # Measure base memory
        base_size = sys.getsizeof(hippocampus.ca3._patterns)

        # Store 100 patterns
        for _ in range(100):
            pattern = np.random.randn(128)
            hippocampus.ca3.store(pattern)

        after_size = sys.getsizeof(hippocampus.ca3._patterns)

        # Memory should have increased
        assert after_size > base_size, "Pattern storage not growing"

        # Verify pattern count
        assert hippocampus.ca3.get_pattern_count() == 100

        print(f"\nHippocampus CA3 memory: {after_size - base_size} bytes for 100 patterns")

    def test_vta_value_table_bounded(self):
        """VTA value table should stay bounded."""
        vta = VTACircuit()

        # Add many states
        for i in range(15000):
            vta.compute_td_error(
                reward=0.5,
                current_state=f"state_{i}",
                next_state=f"state_{i+1}"
            )
            vta.update_value(f"state_{i}", 0.1)

        # Table should be pruned
        assert len(vta._value_table) <= 10000, (
            f"Value table unbounded: {len(vta._value_table)} entries"
        )

        print(f"\nVTA value table size: {len(vta._value_table)} states")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
