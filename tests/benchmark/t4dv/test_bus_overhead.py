"""V7.2 Benchmark: bus emit_sync() overhead must be < 1% of forward pass.

A CorticalBlock forward pass takes ~1-5ms on CPU. We target emit_sync()
taking < 10us per event (< 0.2% of a 5ms forward pass with 6 blocks).
"""

import statistics
import time

import pytest

from t4dm.t4dv.bus import ObservationBus
from t4dm.t4dv.events import SpikeEvent, StorageEvent


class TestBusOverhead:
    """Benchmark emit_sync latency."""

    def test_emit_sync_latency(self):
        bus = ObservationBus(maxlen=10_000)
        event = SpikeEvent(block_index=0, firing_rate=0.5, prediction_error=0.1)

        # Warm up
        for _ in range(1000):
            bus.emit_sync(event)
        bus.clear()

        # Measure
        N = 10_000
        times = []
        for _ in range(N):
            start = time.perf_counter_ns()
            bus.emit_sync(event)
            elapsed_ns = time.perf_counter_ns() - start
            times.append(elapsed_ns)

        median_ns = statistics.median(times)
        p99_ns = sorted(times)[int(N * 0.99)]
        mean_ns = statistics.mean(times)

        print(f"\nemit_sync latency (N={N}):")
        print(f"  mean:   {mean_ns / 1000:.2f} us")
        print(f"  median: {median_ns / 1000:.2f} us")
        print(f"  p99:    {p99_ns / 1000:.2f} us")

        # Assert < 50us median (very conservative; typical is < 5us)
        assert median_ns < 50_000, f"emit_sync median {median_ns}ns > 50us"

    def test_emit_sync_with_subscriber(self):
        """Overhead with one subscriber."""
        bus = ObservationBus(maxlen=10_000)
        count = [0]
        bus.subscribe("spike", lambda e: count.__setitem__(0, count[0] + 1))
        event = SpikeEvent(block_index=0, firing_rate=0.5)

        # emit_sync doesn't await subscribers when no loop is set,
        # so subscriber won't fire. But _store() still runs.
        N = 10_000
        start = time.perf_counter()
        for _ in range(N):
            bus.emit_sync(event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        per_event_us = (elapsed_ms * 1000) / N
        print(f"\nemit_sync with subscriber: {per_event_us:.2f} us/event")
        assert per_event_us < 100, f"Per-event {per_event_us:.1f}us > 100us"

    def test_6_block_overhead_under_1_percent(self):
        """Simulate 6 emit_sync calls (one per block). Total < 1% of 5ms."""
        bus = ObservationBus(maxlen=10_000)
        events = [
            SpikeEvent(block_index=i, firing_rate=0.3 + i * 0.1)
            for i in range(6)
        ]

        # Warm up
        for _ in range(500):
            for e in events:
                bus.emit_sync(e)
        bus.clear()

        N = 1000
        times = []
        for _ in range(N):
            start = time.perf_counter_ns()
            for e in events:
                bus.emit_sync(e)
            elapsed_ns = time.perf_counter_ns() - start
            times.append(elapsed_ns)

        median_ns = statistics.median(times)
        # 1% of 5ms = 50us = 50_000ns
        print(f"\n6-block emit overhead: median {median_ns / 1000:.2f} us")
        assert median_ns < 50_000, (
            f"6-block overhead {median_ns / 1000:.1f}us > 50us (1% of 5ms forward pass)"
        )

    def test_storage_event_overhead(self):
        bus = ObservationBus(maxlen=10_000)
        event = StorageEvent(operation="insert", duration_ms=1.0, segment_count=5)

        N = 10_000
        start = time.perf_counter()
        for _ in range(N):
            bus.emit_sync(event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        per_event_us = (elapsed_ms * 1000) / N
        print(f"\nStorageEvent emit_sync: {per_event_us:.2f} us/event")
        assert per_event_us < 50
