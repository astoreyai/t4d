"""
Tests for Checkpoint Manager.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from ww.persistence.checkpoint import (
    CheckpointManager,
    CheckpointConfig,
    Checkpoint,
    CheckpointableMixin,
)


@pytest.fixture
def checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def checkpoint_config(checkpoint_dir):
    """Create checkpoint config."""
    return CheckpointConfig(
        directory=checkpoint_dir,
        interval_seconds=1.0,  # Fast for tests
        operation_threshold=10,
        max_checkpoints=3,
        compression=True,
    )


@dataclass
class MockComponent(CheckpointableMixin):
    """Mock component for testing."""
    weights: list = None
    bias: float = 0.0
    counter: int = 0

    def __post_init__(self):
        if self.weights is None:
            self.weights = [0.1, 0.2, 0.3]

    def _get_state_keys(self) -> list[str]:
        return ["weights", "bias", "counter"]


class TestCheckpoint:
    """Tests for Checkpoint serialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Checkpoint survives serialization roundtrip."""
        checkpoint = Checkpoint(
            lsn=12345,
            timestamp=1733680000.0,
            gate_state={"W1": [[0.1, 0.2], [0.3, 0.4]]},
            scorer_state={"b1": [0.5, 0.6]},
            buffer_state={"items": [{"id": "1"}, {"id": "2"}]},
            neuromod_state={"dopamine": {"baseline": 0.3}},
        )

        data = checkpoint.serialize(compress=True)
        recovered = Checkpoint.deserialize(data)

        assert recovered.lsn == checkpoint.lsn
        assert recovered.timestamp == checkpoint.timestamp
        assert recovered.gate_state == checkpoint.gate_state
        assert recovered.scorer_state == checkpoint.scorer_state
        assert recovered.buffer_state == checkpoint.buffer_state
        assert recovered.neuromod_state == checkpoint.neuromod_state

    def test_serialize_without_compression(self):
        """Serialization works without compression."""
        checkpoint = Checkpoint(
            lsn=100,
            timestamp=1000.0,
            gate_state={"test": "data"},
        )

        data = checkpoint.serialize(compress=False)
        recovered = Checkpoint.deserialize(data)

        assert recovered.lsn == checkpoint.lsn

    def test_detects_checksum_mismatch(self):
        """Corrupted checkpoint raises ValueError."""
        checkpoint = Checkpoint(lsn=1, timestamp=1.0)
        data = bytearray(checkpoint.serialize(compress=False))

        # Corrupt data (not the header)
        data[-10] ^= 0xFF

        with pytest.raises(ValueError, match="signature mismatch"):
            Checkpoint.deserialize(bytes(data))

    def test_custom_states(self):
        """Custom component states are preserved."""
        checkpoint = Checkpoint(
            lsn=1,
            timestamp=1.0,
            custom_states={
                "my_component": {"field1": 42, "field2": [1, 2, 3]},
                "another": {"nested": {"deep": "value"}},
            },
        )

        data = checkpoint.serialize()
        recovered = Checkpoint.deserialize(data)

        assert recovered.custom_states["my_component"]["field1"] == 42
        assert recovered.custom_states["another"]["nested"]["deep"] == "value"


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.mark.asyncio
    async def test_register_component(self, checkpoint_config):
        """Can register components."""
        manager = CheckpointManager(checkpoint_config)
        component = MockComponent()

        manager.register_component("test", component)

        # Should be tracked
        assert "test" in manager._components

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, checkpoint_config):
        """Can create checkpoint with registered components."""
        manager = CheckpointManager(checkpoint_config)

        component = MockComponent(weights=[1.0, 2.0], bias=0.5, counter=42)
        manager.register_component("test", component)

        await manager.start()
        checkpoint = await manager.create_checkpoint(100)
        await manager.stop()

        assert checkpoint.lsn == 100
        assert checkpoint.custom_states["test"]["weights"] == [1.0, 2.0]
        assert checkpoint.custom_states["test"]["bias"] == 0.5
        assert checkpoint.custom_states["test"]["counter"] == 42

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint(self, checkpoint_config):
        """Can load most recent checkpoint."""
        manager = CheckpointManager(checkpoint_config)
        manager.register_component("test", MockComponent())

        await manager.start()

        # Create several checkpoints
        await manager.create_checkpoint(100)
        await manager.create_checkpoint(200)
        await manager.create_checkpoint(300)

        await manager.stop()

        # Load latest
        loaded = await manager.load_latest_checkpoint()

        assert loaded is not None
        assert loaded.lsn == 300

    @pytest.mark.asyncio
    async def test_restore_all(self, checkpoint_config):
        """Can restore component state from checkpoint."""
        manager = CheckpointManager(checkpoint_config)

        # Create checkpoint with specific state
        original = MockComponent(weights=[9.0, 8.0, 7.0], bias=3.14, counter=999)
        manager.register_component("test", original)

        await manager.start()
        await manager.create_checkpoint(500)
        await manager.stop()

        # Create new component with different state
        restored = MockComponent(weights=[0.0, 0.0, 0.0], bias=0.0, counter=0)
        manager2 = CheckpointManager(checkpoint_config)
        manager2.register_component("test", restored)

        # Load and restore
        checkpoint = await manager2.load_latest_checkpoint()
        results = manager2.restore_all(checkpoint)

        assert results["test"] is True
        assert restored.weights == [9.0, 8.0, 7.0]
        assert restored.bias == 3.14
        assert restored.counter == 999

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self, checkpoint_config):
        """Old checkpoints are cleaned up."""
        checkpoint_config.max_checkpoints = 2
        manager = CheckpointManager(checkpoint_config)
        manager.register_component("test", MockComponent())

        await manager.start()

        # Create more than max
        await manager.create_checkpoint(100)
        await manager.create_checkpoint(200)
        await manager.create_checkpoint(300)
        await manager.create_checkpoint(400)

        await manager.stop()

        # Should only have 2 checkpoints
        checkpoints = manager._list_checkpoints()
        assert len(checkpoints) == 2
        assert 100 not in checkpoints
        assert 200 not in checkpoints
        assert 300 in checkpoints
        assert 400 in checkpoints

    @pytest.mark.asyncio
    async def test_atomic_write_survives_crash(self, checkpoint_config):
        """Checkpoint write is atomic (no partial files)."""
        manager = CheckpointManager(checkpoint_config)
        manager.register_component("test", MockComponent(weights=list(range(1000))))

        await manager.start()
        await manager.create_checkpoint(100)
        await manager.stop()

        # Verify file exists and is valid
        checkpoint = await manager.load_latest_checkpoint()
        assert checkpoint is not None
        assert checkpoint.lsn == 100

        # No temp files should remain
        temp_files = list(checkpoint_config.directory.glob("*.tmp"))
        assert len(temp_files) == 0


class TestCheckpointableMixin:
    """Tests for CheckpointableMixin."""

    def test_get_checkpoint_state(self):
        """Mixin extracts specified state keys."""
        component = MockComponent(weights=[1, 2, 3], bias=0.5, counter=10)
        state = component.get_checkpoint_state()

        assert state["weights"] == [1, 2, 3]
        assert state["bias"] == 0.5
        assert state["counter"] == 10

    def test_restore_from_checkpoint(self):
        """Mixin restores state from dict."""
        component = MockComponent()
        state = {"weights": [9, 9, 9], "bias": 9.9, "counter": 999}

        component.restore_from_checkpoint(state)

        assert component.weights == [9, 9, 9]
        assert component.bias == 9.9
        assert component.counter == 999
