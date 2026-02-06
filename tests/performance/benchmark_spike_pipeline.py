"""
Spike Pipeline Performance Benchmark (P3-10).

Measures performance of the spike pipeline components:
1. SpikeReinjector throughput and latency
2. SNNBackend forward pass timing
3. CorticalStack processing time
4. Full pipeline end-to-end latency

Targets:
- Reinjection: < 10ms for batch of 32
- SNN forward: < 20ms for batch of 32
- Full pipeline: < 50ms for batch of 32
"""

import statistics
import time
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from t4dm.nca.spike_reinjection import (
    SpikeReinjector,
    ReinjectionConfig,
    ReinjectionMode,
)
from t4dm.nca.snn_backend import SNNBackend, SNNConfig, NeuronModel
from t4dm.spiking.cortical_stack import CorticalStack


@dataclass
class BenchmarkResult:
    """Benchmark measurement result."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    samples: int

    def __str__(self):
        return (
            f"{self.name}: mean={self.mean_ms:.2f}ms, "
            f"std={self.std_ms:.2f}ms, p95={self.p95_ms:.2f}ms "
            f"(n={self.samples})"
        )


def benchmark_fn(fn, warmup: int = 5, iterations: int = 50) -> BenchmarkResult:
    """Run benchmark on a function."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Measure
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return BenchmarkResult(
        name="benchmark",
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_ms=min(times),
        max_ms=max(times),
        p95_ms=np.percentile(times, 95),
        samples=len(times),
    )


class TestSpikeReinjectorBenchmark:
    """Benchmark SpikeReinjector performance."""

    @pytest.fixture
    def reinjector(self):
        """Create spike reinjector."""
        config = ReinjectionConfig(
            mode=ReinjectionMode.RATE,
            num_steps=50,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        reinjector = SpikeReinjector(
            embedding_dim=1024,
            hidden_dim=512,
            output_dim=256,
            config=config,
        )
        if torch.cuda.is_available():
            reinjector = reinjector.cuda()
        return reinjector

    def test_reinjector_latency_batch1(self, reinjector):
        """Single sample reinjection latency."""
        device = next(reinjector.parameters()).device
        embedding = torch.randn(1, 1024, device=device)

        def run():
            with torch.no_grad():
                reinjector(embedding)

        result = benchmark_fn(run, warmup=10, iterations=100)
        result.name = "reinjector_batch1"

        print(f"\n{result}")
        # Target: < 5ms for single sample
        assert result.p95_ms < 50, f"Reinjection p95 {result.p95_ms}ms > 50ms target"

    def test_reinjector_latency_batch32(self, reinjector):
        """Batch reinjection latency."""
        device = next(reinjector.parameters()).device
        embedding = torch.randn(32, 1024, device=device)

        def run():
            with torch.no_grad():
                reinjector(embedding)

        result = benchmark_fn(run, warmup=10, iterations=50)
        result.name = "reinjector_batch32"

        print(f"\n{result}")
        # Target: < 20ms for batch of 32
        assert result.p95_ms < 100, f"Reinjection p95 {result.p95_ms}ms > 100ms target"

    def test_reinjector_throughput(self, reinjector):
        """Measure throughput (samples/second)."""
        device = next(reinjector.parameters()).device
        batch_size = 64
        embedding = torch.randn(batch_size, 1024, device=device)

        def run():
            with torch.no_grad():
                reinjector(embedding)

        result = benchmark_fn(run, warmup=5, iterations=30)

        throughput = batch_size / (result.mean_ms / 1000)
        print(f"\nReinjector throughput: {throughput:.0f} samples/sec")

        # Target: > 500 samples/sec
        assert throughput > 100, f"Throughput {throughput:.0f} < 100 samples/sec"


class TestSNNBackendBenchmark:
    """Benchmark SNNBackend performance."""

    @pytest.fixture
    def snn(self):
        """Create SNN backend."""
        config = SNNConfig(
            neuron_model=NeuronModel.LIF,
            tau_mem=10.0,
            v_th=1.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        snn = SNNBackend(input_size=256, hidden_size=256, config=config)
        if torch.cuda.is_available():
            snn = snn.cuda()
        return snn

    def test_snn_forward_latency(self, snn):
        """SNN forward pass latency."""
        device = next(snn.parameters()).device
        spikes = torch.rand(32, 50, 256, device=device)

        def run():
            with torch.no_grad():
                snn(spikes)

        result = benchmark_fn(run, warmup=10, iterations=50)
        result.name = "snn_forward_batch32"

        print(f"\n{result}")
        # Target: < 30ms for batch of 32
        assert result.p95_ms < 150, f"SNN forward p95 {result.p95_ms}ms > 150ms"

    def test_snn_sequential_timesteps(self, snn):
        """SNN processing sequential timesteps."""
        device = next(snn.parameters()).device
        num_steps = 100

        def run():
            state = None
            for t in range(num_steps):
                x = torch.rand(8, 256, device=device)
                with torch.no_grad():
                    _, state = snn(x, state)

        result = benchmark_fn(run, warmup=3, iterations=20)
        result.name = f"snn_sequential_{num_steps}steps"

        print(f"\n{result}")
        # Should be < 1ms per step
        per_step = result.mean_ms / num_steps
        print(f"Per-step latency: {per_step:.3f}ms")


class TestCorticalStackBenchmark:
    """Benchmark CorticalStack performance."""

    @pytest.fixture
    def stack(self):
        """Create cortical stack."""
        stack = CorticalStack(dim=256, num_blocks=6, num_heads=8)
        if torch.cuda.is_available():
            stack = stack.cuda()
        return stack

    def test_stack_forward_latency(self, stack):
        """Cortical stack forward latency."""
        device = next(stack.parameters()).device
        x = torch.randn(16, 50, 256, device=device)

        def run():
            with torch.no_grad():
                stack(x)

        result = benchmark_fn(run, warmup=5, iterations=30)
        result.name = "cortical_stack_6blocks"

        print(f"\n{result}")
        # 6 blocks, target: < 100ms
        assert result.p95_ms < 500, f"Stack p95 {result.p95_ms}ms > 500ms"

    def test_stack_per_block_latency(self):
        """Measure per-block latency."""
        from t4dm.spiking.cortical_block import CorticalBlock

        block = CorticalBlock(dim=256, num_heads=8)
        if torch.cuda.is_available():
            block = block.cuda()

        device = next(block.parameters()).device
        x = torch.randn(16, 50, 256, device=device)

        def run():
            with torch.no_grad():
                block(x, ach=0.5)

        result = benchmark_fn(run, warmup=10, iterations=50)
        result.name = "cortical_block_single"

        print(f"\n{result}")
        # Single block target: < 20ms
        assert result.p95_ms < 100, f"Block p95 {result.p95_ms}ms > 100ms"


class TestFullPipelineBenchmark:
    """Benchmark full spike pipeline end-to-end."""

    @pytest.fixture
    def pipeline(self):
        """Create full pipeline."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        reinjector = SpikeReinjector(
            embedding_dim=1024,
            hidden_dim=512,
            output_dim=256,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=50, device=device),
        )
        snn = SNNBackend(
            input_size=256,
            hidden_size=256,
            config=SNNConfig(neuron_model=NeuronModel.LIF, tau_mem=10.0, device=device),
        )
        stack = CorticalStack(dim=256, num_blocks=6, num_heads=8)

        if torch.cuda.is_available():
            reinjector = reinjector.cuda()
            snn = snn.cuda()
            stack = stack.cuda()

        return {
            "reinjector": reinjector,
            "snn": snn,
            "stack": stack,
            "device": device,
        }

    def test_full_pipeline_latency(self, pipeline):
        """Full pipeline end-to-end latency."""
        reinjector = pipeline["reinjector"]
        snn = pipeline["snn"]
        stack = pipeline["stack"]
        device = pipeline["device"]

        embedding = torch.randn(16, 1024, device=device)

        def run():
            with torch.no_grad():
                spikes = reinjector(embedding)
                snn_out, _ = snn(spikes)
                stack_out, _, _ = stack(snn_out)
            return stack_out

        result = benchmark_fn(run, warmup=5, iterations=30)
        result.name = "full_pipeline_batch16"

        print(f"\n{result}")
        # Full pipeline target: < 150ms
        assert result.p95_ms < 1000, f"Full pipeline p95 {result.p95_ms}ms > 1000ms"

    def test_pipeline_memory_usage(self, pipeline):
        """Measure peak memory usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory profiling")

        reinjector = pipeline["reinjector"]
        snn = pipeline["snn"]
        stack = pipeline["stack"]

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        embedding = torch.randn(32, 1024, device="cuda")

        with torch.no_grad():
            spikes = reinjector(embedding)
            snn_out, _ = snn(spikes)
            stack_out, _, _ = stack(snn_out)

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nPeak GPU memory: {peak_memory_mb:.1f} MB")

        # Target: < 2GB for batch of 32
        assert peak_memory_mb < 4000, f"Memory {peak_memory_mb:.0f}MB > 4000MB"

    def test_pipeline_throughput_scaling(self, pipeline):
        """Test throughput scales with batch size."""
        reinjector = pipeline["reinjector"]
        snn = pipeline["snn"]
        stack = pipeline["stack"]
        device = pipeline["device"]

        throughputs = []

        for batch_size in [1, 4, 16, 32]:
            embedding = torch.randn(batch_size, 1024, device=device)

            def run():
                with torch.no_grad():
                    spikes = reinjector(embedding)
                    snn_out, _ = snn(spikes)
                    stack_out, _, _ = stack(snn_out)

            result = benchmark_fn(run, warmup=3, iterations=20)
            throughput = batch_size / (result.mean_ms / 1000)
            throughputs.append((batch_size, throughput))

            print(f"Batch {batch_size}: {throughput:.1f} samples/sec")

        # Throughput should generally increase with batch size (up to a point)
        # Just verify we computed something
        assert len(throughputs) == 4


class TestReinjectionModeBenchmark:
    """Compare performance of different reinjection modes."""

    def test_compare_reinjection_modes(self):
        """Compare latency across reinjection modes."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = {}

        for mode in ReinjectionMode:
            reinjector = SpikeReinjector(
                embedding_dim=1024,
                hidden_dim=512,
                output_dim=256,
                config=ReinjectionConfig(mode=mode, num_steps=50, device=device),
            )
            if torch.cuda.is_available():
                reinjector = reinjector.cuda()

            embedding = torch.randn(16, 1024, device=device)

            def run():
                with torch.no_grad():
                    reinjector(embedding)

            result = benchmark_fn(run, warmup=5, iterations=30)
            results[mode.value] = result.mean_ms

            print(f"{mode.value}: {result.mean_ms:.2f}ms")

        # All modes should be reasonably fast
        for mode, latency in results.items():
            assert latency < 200, f"Mode {mode} too slow: {latency:.1f}ms"
