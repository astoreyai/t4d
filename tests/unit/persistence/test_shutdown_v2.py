"""Tests for shutdown v2 (P5-03)."""

import pytest
import time
import uuid

import numpy as np

from t4dm.persistence.shutdown_v2 import ShutdownManagerV2, ShutdownV2Result
from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import ItemRecord


def _make_item(**kwargs):
    defaults = dict(
        id=uuid.uuid4().bytes,
        vector=np.random.randn(32).tolist(),
        event_time=time.time(),
        record_time=time.time(),
        valid_from=time.time(),
        valid_until=None,
        kappa=0.1,
        importance=0.5,
        item_type="episode",
        content="test",
        access_count=0,
        session_id=None,
    )
    defaults.update(kwargs)
    return ItemRecord(**defaults)


class TestShutdownV2:
    def test_clean_shutdown_empty(self, engine):
        sm = ShutdownManagerV2(engine)
        result = sm.execute_shutdown()
        assert result.success
        assert result.memtable_flushed
        assert result.wal_fsynced
        assert result.handles_closed
        assert result.shutdown_marker_written

    def test_clean_shutdown_with_data(self, engine):
        for _ in range(5):
            engine.insert(_make_item())

        sm = ShutdownManagerV2(engine)
        result = sm.execute_shutdown()
        assert result.success
        assert result.memtable_flushed

        # Verify shutdown marker exists
        marker = engine.data_dir / "shutdown_marker"
        assert marker.exists()

    def test_shutdown_with_checkpoint(self, engine):
        engine.insert(_make_item())
        checkpoint_called = []

        def fake_checkpoint():
            checkpoint_called.append(True)

        sm = ShutdownManagerV2(engine, checkpoint_fn=fake_checkpoint)
        result = sm.execute_shutdown()
        assert result.success
        assert result.checkpoint_created
        assert len(checkpoint_called) == 1

    def test_accepting_writes_flag(self, engine):
        sm = ShutdownManagerV2(engine)
        assert sm.accepting_writes
        sm.execute_shutdown()
        assert not sm.accepting_writes

    def test_segments_cleared(self, engine):
        # Flush to create a segment
        for _ in range(5):
            engine.insert(_make_item())
        engine.flush()
        assert engine.segment_count > 0

        sm = ShutdownManagerV2(engine)
        result = sm.execute_shutdown()
        assert result.success
        assert result.handles_closed
        assert engine.segment_count == 0  # cleared
