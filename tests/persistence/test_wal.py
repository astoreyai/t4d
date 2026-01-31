"""
Tests for Write-Ahead Log (WAL) system.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from t4dm.persistence.wal import (
    WriteAheadLog,
    WALConfig,
    WALEntry,
    WALOperation,
    WALCorruptionError,
)


@pytest.fixture
def wal_dir():
    """Create temporary WAL directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def wal_config(wal_dir):
    """Create WAL config."""
    return WALConfig(
        directory=wal_dir,
        segment_max_size=1024 * 1024,  # 1MB for tests
        sync_mode="none",  # Fast for tests
    )


@pytest.fixture
async def wal(wal_config):
    """Create and open WAL."""
    wal = WriteAheadLog(wal_config)
    await wal.open()
    yield wal
    await wal.close()


class TestWALEntry:
    """Tests for WALEntry serialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Entry survives serialization roundtrip."""
        entry = WALEntry(
            lsn=42,
            timestamp_ns=1733680000000000000,
            operation=WALOperation.BUFFER_ADD,
            payload={"memory_id": "test_123", "content": "Hello world"},
        )

        data = entry.serialize()
        recovered, consumed = WALEntry.deserialize(data)

        assert recovered.lsn == entry.lsn
        assert recovered.timestamp_ns == entry.timestamp_ns
        assert recovered.operation == entry.operation
        assert recovered.payload == entry.payload
        assert consumed == len(data)

    def test_detects_corruption(self):
        """Corrupted data raises WALCorruptionError."""
        entry = WALEntry(
            lsn=1,
            timestamp_ns=1000,
            operation=WALOperation.BUFFER_ADD,
            payload={"test": "data"},
        )

        data = bytearray(entry.serialize())
        # Corrupt the checksum/HMAC
        data[-1] ^= 0xFF

        with pytest.raises(WALCorruptionError, match="Integrity check failed"):
            WALEntry.deserialize(bytes(data))

    def test_detects_invalid_magic(self):
        """Invalid magic number raises WALCorruptionError."""
        data = b'\x00\x00\x00\x00' + b'\x00' * 50

        with pytest.raises(WALCorruptionError, match="Invalid magic"):
            WALEntry.deserialize(data)

    def test_handles_complex_payload(self):
        """Complex nested payloads serialize correctly."""
        entry = WALEntry(
            lsn=100,
            timestamp_ns=2000,
            operation=WALOperation.GATE_WEIGHT_UPDATE,
            payload={
                "layer": "W1",
                "weights": [[0.1, 0.2], [0.3, 0.4]],
                "metadata": {"lr": 0.01, "epoch": 5},
            },
        )

        data = entry.serialize()
        recovered, _ = WALEntry.deserialize(data)

        assert recovered.payload["weights"] == [[0.1, 0.2], [0.3, 0.4]]
        assert recovered.payload["metadata"]["lr"] == 0.01


class TestWriteAheadLog:
    """Tests for WriteAheadLog."""

    @pytest.mark.asyncio
    async def test_open_creates_directory(self, wal_config):
        """WAL creates directory if not exists."""
        new_dir = wal_config.directory / "subdir"
        wal_config.directory = new_dir

        wal = WriteAheadLog(wal_config)
        await wal.open()

        assert new_dir.exists()
        await wal.close()

    @pytest.mark.asyncio
    async def test_append_returns_lsn(self, wal):
        """Append returns monotonically increasing LSN."""
        lsn1 = await wal.append(WALOperation.BUFFER_ADD, {"id": "1"})
        lsn2 = await wal.append(WALOperation.BUFFER_ADD, {"id": "2"})
        lsn3 = await wal.append(WALOperation.BUFFER_ADD, {"id": "3"})

        assert lsn2 == lsn1 + 1
        assert lsn3 == lsn2 + 1

    @pytest.mark.asyncio
    async def test_iter_entries(self, wal):
        """Can iterate all entries."""
        payloads = [{"id": str(i)} for i in range(10)]

        for payload in payloads:
            await wal.append(WALOperation.BUFFER_ADD, payload)

        recovered = []
        async for entry in wal.iter_entries():
            if entry.operation == WALOperation.BUFFER_ADD:
                recovered.append(entry.payload)

        assert recovered == payloads

    @pytest.mark.asyncio
    async def test_iter_entries_with_filter(self, wal):
        """Can filter entries by operation type."""
        await wal.append(WALOperation.BUFFER_ADD, {"type": "add"})
        await wal.append(WALOperation.BUFFER_REMOVE, {"type": "remove"})
        await wal.append(WALOperation.BUFFER_ADD, {"type": "add2"})

        adds = []
        async for entry in wal.iter_entries(operations={WALOperation.BUFFER_ADD}):
            adds.append(entry.payload)

        assert len(adds) == 2
        assert adds[0]["type"] == "add"
        assert adds[1]["type"] == "add2"

    @pytest.mark.asyncio
    async def test_iter_entries_from_lsn(self, wal):
        """Can start iteration from specific LSN."""
        lsns = []
        for i in range(5):
            lsn = await wal.append(WALOperation.BUFFER_ADD, {"i": i})
            lsns.append(lsn)

        # Start from lsn[2]
        recovered = []
        async for entry in wal.iter_entries(from_lsn=lsns[2]):
            if entry.operation == WALOperation.BUFFER_ADD:
                recovered.append(entry.payload["i"])

        assert recovered == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_recovery_after_close(self, wal_config):
        """WAL recovers state after close and reopen."""
        # First session
        wal1 = WriteAheadLog(wal_config)
        await wal1.open()

        await wal1.append(WALOperation.BUFFER_ADD, {"session": 1, "i": 0})
        await wal1.append(WALOperation.BUFFER_ADD, {"session": 1, "i": 1})
        last_lsn = await wal1.append(WALOperation.BUFFER_ADD, {"session": 1, "i": 2})

        await wal1.close()

        # Second session
        wal2 = WriteAheadLog(wal_config)
        await wal2.open()

        # LSN should continue from where we left off
        # After close: SYSTEM_SHUTDOWN (+1), on open: SYSTEM_START (+1), new entry (+1)
        new_lsn = await wal2.append(WALOperation.BUFFER_ADD, {"session": 2, "i": 0})
        assert new_lsn == last_lsn + 3  # +1 for SYSTEM_SHUTDOWN, +1 for SYSTEM_START, +1 for new entry

        # Should see all entries
        count = 0
        async for entry in wal2.iter_entries():
            if entry.operation == WALOperation.BUFFER_ADD:
                count += 1

        assert count == 4  # 3 from session 1 + 1 from session 2

        await wal2.close()

    @pytest.mark.asyncio
    async def test_checkpoint_marker(self, wal):
        """Checkpoint marker is recorded."""
        await wal.append(WALOperation.BUFFER_ADD, {"i": 0})
        await wal.append(WALOperation.BUFFER_ADD, {"i": 1})

        checkpoint_lsn = wal.current_lsn - 1
        await wal.mark_checkpoint(checkpoint_lsn)

        assert wal.checkpoint_lsn == checkpoint_lsn

    @pytest.mark.asyncio
    async def test_iter_uncommitted(self, wal):
        """iter_uncommitted returns entries after checkpoint."""
        # Write some entries
        await wal.append(WALOperation.BUFFER_ADD, {"before": True})
        await wal.append(WALOperation.BUFFER_ADD, {"before": True})
        checkpoint_lsn = wal.current_lsn - 1

        await wal.mark_checkpoint(checkpoint_lsn)

        await wal.append(WALOperation.BUFFER_ADD, {"after": True})
        await wal.append(WALOperation.BUFFER_ADD, {"after": True})

        uncommitted = []
        async for entry in wal.iter_uncommitted(checkpoint_lsn):
            if entry.operation == WALOperation.BUFFER_ADD:
                uncommitted.append(entry.payload)

        assert len(uncommitted) == 2
        assert all(p.get("after") for p in uncommitted)


class TestWALSegmentRotation:
    """Tests for WAL segment rotation."""

    @pytest.mark.asyncio
    async def test_rotates_at_size_limit(self, wal_dir):
        """WAL rotates to new segment when size limit reached."""
        config = WALConfig(
            directory=wal_dir,
            segment_max_size=1024,  # 1KB - very small for test
            sync_mode="none",
        )

        wal = WriteAheadLog(config)
        await wal.open()

        # Write until we force rotation
        large_payload = {"data": "x" * 200}
        for _ in range(20):
            await wal.append(WALOperation.BUFFER_ADD, large_payload)

        await wal.close()

        # Should have multiple segments
        segments = list(wal_dir.glob("segment_*.wal"))
        assert len(segments) > 1

    @pytest.mark.asyncio
    async def test_reads_across_segments(self, wal_dir):
        """Can read entries across multiple segments."""
        config = WALConfig(
            directory=wal_dir,
            segment_max_size=512,
            sync_mode="none",
        )

        wal = WriteAheadLog(config)
        await wal.open()

        # Write many entries
        expected = []
        for i in range(50):
            payload = {"index": i, "data": f"entry_{i}"}
            await wal.append(WALOperation.BUFFER_ADD, payload)
            expected.append(payload)

        await wal.close()

        # Reopen and read all
        wal2 = WriteAheadLog(config)
        await wal2.open()

        recovered = []
        async for entry in wal2.iter_entries():
            if entry.operation == WALOperation.BUFFER_ADD:
                recovered.append(entry.payload)

        await wal2.close()

        assert recovered == expected
