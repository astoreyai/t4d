"""Tests for T4DV FastAPI server."""

import pytest
from fastapi.testclient import TestClient

from ww.t4dv.aggregator import SnapshotAggregator
from ww.t4dv.bus import ObservationBus
from ww.t4dv.events import SpikeEvent
from ww.t4dv.server import create_app


@pytest.fixture
def client():
    bus = ObservationBus()
    agg = SnapshotAggregator()
    app = create_app(bus=bus, aggregator=agg)
    # Pre-populate
    event = SpikeEvent(block_index=0, firing_rate=0.5)
    bus.emit_sync(event)
    return TestClient(app), bus, agg


class TestRestEndpoints:
    def test_dashboard(self, client):
        tc, bus, agg = client
        # Ingest into aggregator
        agg.ingest(SpikeEvent(block_index=0, firing_rate=0.5))
        resp = tc.get("/api/v1/viz/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "spiking" in data

    def test_snapshot(self, client):
        tc, bus, agg = client
        resp = tc.get("/api/v1/viz/snapshot/spike")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["topic"] == "spike"

    def test_topics(self, client):
        tc, bus, agg = client
        resp = tc.get("/api/v1/viz/topics")
        assert resp.status_code == 200
        assert "spike" in resp.json()

    def test_views(self, client):
        tc, bus, agg = client
        resp = tc.get("/api/v1/viz/views")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
