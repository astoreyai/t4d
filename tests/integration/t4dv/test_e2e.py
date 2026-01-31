"""V7.1 E2E: bus → aggregator → server → client receives events."""

import pytest
from fastapi.testclient import TestClient

from t4dm.t4dv.aggregator import SnapshotAggregator
from t4dm.t4dv.bus import ObservationBus
from t4dm.t4dv.events import ConsolidationEvent, NeuromodEvent, SpikeEvent, StorageEvent
from t4dm.t4dv.server import create_app


def _emit(bus: ObservationBus, agg: SnapshotAggregator, event):
    """Emit into bus and ingest into aggregator (simulates async subscribe)."""
    bus.emit_sync(event)
    agg.ingest(event)


@pytest.fixture
def stack():
    bus = ObservationBus()
    agg = SnapshotAggregator()
    app = create_app(bus=bus, aggregator=agg)
    return bus, agg, TestClient(app)


class TestE2EPipeline:
    """End-to-end: emit events → aggregator → REST dashboard returns data."""

    def test_spike_to_dashboard(self, stack):
        bus, agg, client = stack
        for i in range(5):
            _emit(bus, agg, SpikeEvent(block_index=i % 3, firing_rate=0.1 * i))

        resp = client.get("/api/v1/viz/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spiking"]["windows"]["1s"]["count"] == 5

    def test_storage_to_dashboard(self, stack):
        bus, agg, client = stack
        for op in ["insert", "get", "search", "insert"]:
            _emit(bus, agg, StorageEvent(operation=op, duration_ms=1.0, segment_count=2))

        resp = client.get("/api/v1/viz/dashboard")
        data = resp.json()
        assert data["storage"]["windows"]["1s"]["ops_total"] == 4
        assert data["storage"]["windows"]["1s"]["ops_by_type"]["insert"] == 2

    def test_consolidation_to_dashboard(self, stack):
        bus, agg, client = stack
        _emit(bus, agg, ConsolidationEvent(phase="nrem", items_processed=10))
        _emit(bus, agg, ConsolidationEvent(phase="rem", items_processed=3))

        resp = client.get("/api/v1/viz/dashboard")
        data = resp.json()
        timeline = data["consolidation"]["recent_60s"]
        assert len(timeline) == 2
        assert timeline[0]["phase"] == "nrem"

    def test_neuromod_to_dashboard(self, stack):
        bus, agg, client = stack
        _emit(bus, agg, NeuromodEvent(da=0.9, ne=0.2, ach=0.7, serotonin=0.4))

        resp = client.get("/api/v1/viz/dashboard")
        data = resp.json()
        assert data["neuromod"]["latest"]["da"] == 0.9
        assert len(data["neuromod"]["series"]["da"]) == 1

    def test_snapshot_endpoint(self, stack):
        bus, agg, client = stack
        bus.emit_sync(SpikeEvent(block_index=0, firing_rate=0.5))
        bus.emit_sync(SpikeEvent(block_index=1, firing_rate=0.8))

        resp = client.get("/api/v1/viz/snapshot/spike")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 2
        assert events[0]["topic"] == "spike"

    def test_topics_endpoint(self, stack):
        bus, agg, client = stack
        bus.emit_sync(SpikeEvent())
        bus.emit_sync(StorageEvent())

        resp = client.get("/api/v1/viz/topics")
        topics = resp.json()
        assert "spike" in topics
        assert "storage" in topics

    def test_mixed_event_flow(self, stack):
        """Simulate realistic mixed event stream."""
        bus, agg, client = stack

        for block_idx in range(6):
            _emit(bus, agg, SpikeEvent(
                block_index=block_idx,
                firing_rate=0.3 + block_idx * 0.1,
                prediction_error=0.05,
                goodness=1.5,
            ))
        _emit(bus, agg, StorageEvent(
            operation="insert", duration_ms=0.5, segment_count=4, memtable_count=150,
        ))
        _emit(bus, agg, NeuromodEvent(da=0.6, ne=0.4, ach=0.55, serotonin=0.5))

        resp = client.get("/api/v1/viz/dashboard")
        data = resp.json()

        assert data["spiking"]["windows"]["1s"]["count"] == 6
        assert data["storage"]["segment_count"] == 4
        assert data["neuromod"]["latest"]["da"] == 0.6
