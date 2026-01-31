"""Tests for T4DV observation events."""

from datetime import datetime

from ww.t4dv.events import (
    ConsolidationEvent,
    NeuromodEvent,
    ObservationEvent,
    SpikeEvent,
    StorageEvent,
)


class TestObservationEvent:
    def test_base_event_defaults(self):
        e = ObservationEvent(topic="test")
        assert e.topic == "test"
        assert isinstance(e.timestamp, datetime)
        assert e.source == ""
        assert e.payload == {}

    def test_base_event_custom(self):
        e = ObservationEvent(topic="x", source="src", payload={"k": 1})
        assert e.source == "src"
        assert e.payload["k"] == 1


class TestSpikeEvent:
    def test_defaults(self):
        e = SpikeEvent()
        assert e.topic == "spike"
        assert e.block_index == 0
        assert e.firing_rate == 0.0
        assert e.prediction_error == 0.0
        assert e.goodness == 0.0

    def test_custom(self):
        e = SpikeEvent(block_index=3, firing_rate=0.7, prediction_error=0.1, goodness=2.5)
        assert e.block_index == 3
        assert e.firing_rate == 0.7

    def test_serialization(self):
        e = SpikeEvent(block_index=1, firing_rate=0.5)
        d = e.model_dump()
        assert d["topic"] == "spike"
        assert d["block_index"] == 1
        e2 = SpikeEvent.model_validate(d)
        assert e2.block_index == 1


class TestStorageEvent:
    def test_defaults(self):
        e = StorageEvent()
        assert e.topic == "storage"
        assert e.operation == ""

    def test_fields(self):
        e = StorageEvent(operation="insert", item_id="abc", kappa=0.3, duration_ms=1.5)
        assert e.operation == "insert"
        assert e.kappa == 0.3


class TestConsolidationEvent:
    def test_phases(self):
        for phase in ["nrem", "rem", "prune"]:
            e = ConsolidationEvent(phase=phase)
            assert e.phase == phase
            assert e.topic == "consolidation"


class TestNeuromodEvent:
    def test_defaults(self):
        e = NeuromodEvent()
        assert e.topic == "neuromod"
        assert e.da == 0.5
        assert e.ach == 0.5
