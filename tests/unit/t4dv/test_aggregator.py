"""Tests for SnapshotAggregator."""

from t4dm.t4dv.aggregator import SnapshotAggregator
from t4dm.t4dv.events import NeuromodEvent, SpikeEvent, StorageEvent


class TestSnapshotAggregator:
    def test_empty_dashboard(self):
        agg = SnapshotAggregator()
        state = agg.get_dashboard_state()
        assert "spiking" in state
        assert "storage" in state
        assert state["spiking"]["windows"]["1s"]["count"] == 0

    def test_ingest_spike(self):
        agg = SnapshotAggregator()
        agg.ingest(SpikeEvent(block_index=0, firing_rate=0.5))
        agg.ingest(SpikeEvent(block_index=1, firing_rate=0.8))
        state = agg.get_dashboard_state()
        assert state["spiking"]["windows"]["1s"]["count"] == 2
        assert abs(state["spiking"]["windows"]["1s"]["mean_firing_rate"] - 0.65) < 0.01

    def test_ingest_storage(self):
        agg = SnapshotAggregator()
        agg.ingest(StorageEvent(operation="insert", duration_ms=1.0, segment_count=3))
        agg.ingest(StorageEvent(operation="get", duration_ms=0.5, segment_count=3))
        state = agg.get_dashboard_state()
        assert state["storage"]["windows"]["1s"]["ops_total"] == 2
        assert state["storage"]["segment_count"] == 3

    def test_ingest_neuromod(self):
        agg = SnapshotAggregator()
        agg.ingest(NeuromodEvent(da=0.9, ne=0.1))
        state = agg.get_dashboard_state()
        assert state["neuromod"]["latest"]["da"] == 0.9
        assert len(state["neuromod"]["series"]["da"]) == 1
