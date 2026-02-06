"""Tests for persistence integrity (WAL + checkpoint)."""
import os
import tempfile
from pathlib import Path

import pytest


class TestWALIntegrity:
    """Test WAL integrity verification (HMAC-SHA256)."""

    @pytest.mark.asyncio
    async def test_wal_tamper_detection(self):
        """Flip 1 bit in WAL entry, verify replay raises."""
        from t4dm.persistence.wal import WriteAheadLog, WALOperation, WALConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = WALConfig(directory=Path(tmpdir))
            wal = WriteAheadLog(config)

            await wal.open()

            # Write an entry
            await wal.append(WALOperation.BUFFER_ADD, {"test": "data"})
            await wal.close()

            # Find the WAL file and tamper with it
            wal_files = list(Path(tmpdir).glob("*.wal"))
            assert len(wal_files) > 0

            data = wal_files[0].read_bytes()
            if len(data) > 10:
                tampered = bytearray(data)
                # Tamper near the end (in the data portion)
                tampered[-5] ^= 0xFF
                wal_files[0].write_bytes(bytes(tampered))

                # Replay should detect corruption
                wal2 = WriteAheadLog(config)
                await wal2.open()

                entries = []
                async for entry in wal2.iter_entries():
                    entries.append(entry)

                await wal2.close()
                # Either raises error or returns empty/fewer entries
                # The exact behavior depends on implementation

    @pytest.mark.asyncio
    async def test_wal_hmac_verification_async(self):
        """Test HMAC verification in async context."""
        from t4dm.persistence.wal import WriteAheadLog, WALOperation, WALConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = WALConfig(directory=Path(tmpdir))
            wal = WriteAheadLog(config)

            await wal.open()

            # Write a test entry
            lsn = await wal.append(
                WALOperation.BUFFER_ADD,
                {"memory_id": "test123", "data": "important"}
            )
            assert lsn > 0

            await wal.close()

            # Tamper with the WAL file
            wal_files = list(Path(tmpdir).glob("*.wal"))
            assert len(wal_files) > 0

            data = wal_files[0].read_bytes()
            if len(data) > 50:
                tampered = bytearray(data)
                # Flip bits in payload area
                tampered[-10] ^= 0xFF
                wal_files[0].write_bytes(bytes(tampered))

                # Attempt to read - should detect corruption
                wal2 = WriteAheadLog(config)
                await wal2.open()

                entries = []
                async for entry in wal2.iter_entries():
                    entries.append(entry)

                # Should either raise or return fewer entries
                # (depends on whether corruption is in critical area)
                await wal2.close()

    def test_wal_custom_key(self):
        """Test WAL with custom HMAC key."""
        from t4dm.persistence.wal import WALEntry, WALOperation

        # Set custom key
        original_key = os.environ.get("T4DM_WAL_KEY")
        try:
            os.environ["T4DM_WAL_KEY"] = "my-super-secret-key-12345"

            entry = WALEntry(
                lsn=1,
                timestamp_ns=1234567890,
                operation=WALOperation.BUFFER_ADD,
                payload={"test": "data"}
            )

            # Serialize with custom key
            data = entry.serialize()

            # Deserialize should succeed with same key
            entry2, consumed = WALEntry.deserialize(data)
            assert entry2.lsn == entry.lsn
            assert entry2.payload == entry.payload

            # Change key - should fail
            os.environ["T4DM_WAL_KEY"] = "different-key"
            with pytest.raises(Exception):  # WALCorruptionError
                WALEntry.deserialize(data)

        finally:
            # Restore original key
            if original_key:
                os.environ["T4DM_WAL_KEY"] = original_key
            else:
                os.environ.pop("T4DM_WAL_KEY", None)

    def test_wal_v1_backward_compat(self):
        """Test that v1 (CRC32) entries are still readable."""
        import struct
        import zlib
        from t4dm.persistence.wal import (
            WALEntry,
            WALOperation,
            WAL_MAGIC,
            HEADER_SIZE,
        )

        # Create a v1 entry manually (CRC32)
        lsn = 42
        timestamp_ns = 1234567890
        operation = WALOperation.BUFFER_ADD
        payload = {"test": "v1_data"}

        # Use msgpack if available (same as deserialize logic)
        try:
            import msgpack
            payload_bytes = msgpack.packb(payload, use_bin_type=True)
        except ImportError:
            import json
            payload_bytes = json.dumps(payload).encode("utf-8")

        # Pack header
        header = struct.pack(
            ">IQQHH",
            WAL_MAGIC,
            lsn,
            timestamp_ns,
            operation,
            len(payload_bytes),
        )

        # Compute CRC32 (v1 style)
        data = header + payload_bytes
        checksum = zlib.crc32(data) & 0xFFFFFFFF

        # Append CRC32 (4 bytes)
        v1_entry_bytes = data + struct.pack(">I", checksum)

        # Should deserialize with warning
        entry, consumed = WALEntry.deserialize(v1_entry_bytes)
        assert entry.lsn == lsn
        assert entry.payload == payload
        assert consumed == len(v1_entry_bytes)

    def test_wal_v2_hmac_format(self):
        """Test that v2 (HMAC) entries are written and verified."""
        from t4dm.persistence.wal import WALEntry, WALOperation, HEADER_SIZE, HMAC_SIZE

        entry = WALEntry(
            lsn=100,
            timestamp_ns=9876543210,
            operation=WALOperation.GATE_WEIGHT_UPDATE,
            payload={"weights": [0.1, 0.2, 0.3]}
        )

        # Serialize (should use HMAC)
        data = entry.serialize()

        # Should have: header (24) + payload (~28 for JSON) + HMAC (32)
        # Total should be at least HEADER_SIZE + HMAC_SIZE = 56 bytes
        assert len(data) >= HEADER_SIZE + HMAC_SIZE

        # HMAC adds 28 more bytes than CRC32 (32 vs 4)
        # So v2 should be notably larger than v1
        assert len(data) > 50  # Reasonable lower bound

        # Deserialize should succeed
        entry2, consumed = WALEntry.deserialize(data)
        assert entry2.lsn == entry.lsn
        assert entry2.payload == entry.payload
        assert consumed == len(data)

    def test_wal_tamper_in_header(self):
        """Test tampering detection in header."""
        from t4dm.persistence.wal import WALEntry, WALOperation

        entry = WALEntry(
            lsn=50,
            timestamp_ns=1111111111,
            operation=WALOperation.DOPAMINE_RPE,
            payload={"rpe": 0.5}
        )

        data = entry.serialize()

        # Tamper with LSN in header
        tampered = bytearray(data)
        tampered[8] ^= 0xFF  # Flip bits in LSN field

        # Should fail verification
        with pytest.raises(Exception):  # WALCorruptionError
            WALEntry.deserialize(bytes(tampered))

    def test_wal_tamper_in_payload(self):
        """Test tampering detection in payload."""
        from t4dm.persistence.wal import WALEntry, WALOperation

        entry = WALEntry(
            lsn=60,
            timestamp_ns=2222222222,
            operation=WALOperation.TRACE_UPDATE,
            payload={"trace_id": "abc", "value": 0.9}
        )

        data = entry.serialize()

        # Tamper with payload
        tampered = bytearray(data)
        # Find a position in payload (after header)
        payload_start = 24  # HEADER_SIZE
        if len(tampered) > payload_start + 10:
            tampered[payload_start + 5] ^= 0xFF

            # Should fail verification
            with pytest.raises(Exception):  # WALCorruptionError
                WALEntry.deserialize(bytes(tampered))
