"""Tests for emitters (spiking, storage, consolidation, neuromod)."""

from unittest.mock import MagicMock

import pytest

from ww.t4dv.bus import ObservationBus
from ww.t4dv.emitters.consolidation import attach_consolidation_hooks
from ww.t4dv.emitters.neuromod import sample_neuromod
from ww.t4dv.emitters.storage import attach_storage_hooks


class TestStorageEmitter:
    def test_wraps_ops(self):
        bus = ObservationBus()
        engine = MagicMock()
        engine.insert = MagicMock(return_value=None)
        engine.get = MagicMock(return_value=None)
        engine.segment_count = 2
        engine.memtable_count = 5

        attach_storage_hooks(engine, bus)

        # Call wrapped insert
        engine.insert("fake_record")
        snap = bus.snapshot("storage")
        assert len(snap) == 1
        assert snap[0].operation == "insert"
        assert snap[0].segment_count == 2
        assert snap[0].duration_ms >= 0

    def test_get_wrapped(self):
        bus = ObservationBus()
        engine = MagicMock()
        engine.get = MagicMock(return_value="result")
        engine.segment_count = 0
        engine.memtable_count = 0

        attach_storage_hooks(engine, bus)
        result = engine.get(b"\x00" * 16)
        assert result == "result"
        snap = bus.snapshot("storage")
        assert len(snap) == 1
        assert snap[0].operation == "get"


class TestConsolidationEmitter:
    def test_wraps_nrem(self):
        bus = ObservationBus()
        engine = MagicMock()
        engine.nrem_compact = MagicMock(return_value=5)
        engine.segment_count = 3

        attach_consolidation_hooks(engine, bus)
        result = engine.nrem_compact()
        assert result == 5
        snap = bus.snapshot("consolidation")
        assert len(snap) == 1
        assert snap[0].phase == "nrem"
        assert snap[0].items_processed == 5


class TestNeuromodEmitter:
    def test_sample(self):
        bus = ObservationBus()
        state = MagicMock()
        state.da = 0.8
        state.ne = 0.3
        state.ach = 0.6
        state.serotonin = 0.4

        sample_neuromod(state, bus)
        snap = bus.snapshot("neuromod")
        assert len(snap) == 1
        assert snap[0].da == 0.8
        assert snap[0].ach == 0.6

    def test_missing_attrs_default(self):
        bus = ObservationBus()
        state = MagicMock(spec=[])  # no attributes

        sample_neuromod(state, bus)
        snap = bus.snapshot("neuromod")
        assert snap[0].da == 0.5
