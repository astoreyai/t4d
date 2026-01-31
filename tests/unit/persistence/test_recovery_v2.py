"""Tests for recovery v2 (P5-02)."""

import pytest
import time
import uuid

import numpy as np

from ww.persistence.checkpoint import CheckpointConfig, CheckpointManager
from ww.persistence.checkpoint_v3 import CheckpointManagerV3, T4DXCheckpointable
from ww.persistence.recovery_v2 import RecoveryManagerV2
from ww.storage.t4dx.engine import T4DXEngine
from ww.storage.t4dx.types import ItemRecord


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


class TestRecoveryV2:
    @pytest.mark.asyncio
    async def test_cold_start(self, data_dir):
        engine = T4DXEngine(data_dir)
        cfg = CheckpointConfig(directory=data_dir / "checkpoints")
        cp_mgr = CheckpointManager(cfg)

        recovery = RecoveryManagerV2(engine, cp_mgr)
        result = await recovery.recover(force_cold=True)

        assert result.success
        assert result.mode == "cold"
        assert result.segments_loaded == 0
        engine.shutdown()

    @pytest.mark.asyncio
    async def test_warm_start(self, data_dir):
        # Phase 1: create data and checkpoint
        engine = T4DXEngine(data_dir)
        engine.startup()
        item = _make_item()
        engine.insert(item)

        cp_dir = data_dir / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        cfg = CheckpointConfig(directory=cp_dir)
        cp_mgr = CheckpointManagerV3(cfg)
        cp_mgr.register_t4dx(engine)
        await cp_mgr.create_checkpoint(lsn=1)
        engine.shutdown()

        # Phase 2: recover
        engine2 = T4DXEngine(data_dir)
        cp_mgr2 = CheckpointManagerV3(cfg)
        cp_mgr2.register_t4dx(engine2)

        recovery = RecoveryManagerV2(engine2, cp_mgr2)
        result = await recovery.recover()

        assert result.success
        assert result.mode == "warm"
        assert result.checkpoint_lsn == 1
        engine2.shutdown()

    @pytest.mark.asyncio
    async def test_cold_start_no_checkpoint(self, data_dir):
        engine = T4DXEngine(data_dir)
        cfg = CheckpointConfig(directory=data_dir / "checkpoints")
        cp_mgr = CheckpointManager(cfg)

        recovery = RecoveryManagerV2(engine, cp_mgr)
        result = await recovery.recover()

        assert result.success
        assert result.mode == "cold"
        engine.shutdown()

    @pytest.mark.asyncio
    async def test_consistency_validator(self, data_dir):
        engine = T4DXEngine(data_dir)
        cfg = CheckpointConfig(directory=data_dir / "checkpoints")
        cp_mgr = CheckpointManager(cfg)

        recovery = RecoveryManagerV2(engine, cp_mgr)
        recovery.register_consistency_validator(lambda: True)
        result = await recovery.recover(force_cold=True)
        assert result.success
        engine.shutdown()

    @pytest.mark.asyncio
    async def test_failed_validator(self, data_dir):
        engine = T4DXEngine(data_dir)
        cfg = CheckpointConfig(directory=data_dir / "checkpoints")
        cp_mgr = CheckpointManager(cfg)

        recovery = RecoveryManagerV2(engine, cp_mgr)
        recovery.register_consistency_validator(lambda: False)
        result = await recovery.recover(force_cold=True)
        assert not result.success
        engine.shutdown()
