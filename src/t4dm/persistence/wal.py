"""
Write-Ahead Log (WAL) for World Weaver

Guarantees durability by logging operations BEFORE they're applied to in-memory state.
On crash, replay uncommitted entries to restore consistency.

WAL Entry Format v2 (binary):
=============================
    [4 bytes] Magic number (0xWW4C)
    [8 bytes] LSN (Log Sequence Number)
    [8 bytes] Timestamp (nanoseconds since epoch)
    [2 bytes] Operation type
    [4 bytes] Payload length
    [N bytes] Payload (msgpack serialized)
    [32 bytes] HMAC-SHA256 (keyed by T4DM_WAL_KEY env var)

WAL Entry Format v1 (legacy, backward compatible):
==================================================
    [4 bytes] Magic number (0xWW4C)
    [8 bytes] LSN (Log Sequence Number)
    [8 bytes] Timestamp (nanoseconds since epoch)
    [2 bytes] Operation type
    [4 bytes] Payload length
    [N bytes] Payload (msgpack serialized)
    [4 bytes] CRC32 checksum (deprecated, insecure)

Segment Files:
=============
    wal/
    ├── segment_00000001.wal  (64MB max)
    ├── segment_00000002.wal
    ├── segment_00000003.wal
    └── current -> segment_00000003.wal

Security:
========
Set T4DM_WAL_KEY environment variable to a strong secret for HMAC verification.
If not set, a default development key is used (NOT SECURE FOR PRODUCTION).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac
import logging
import os
import struct
import threading
import time
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

try:
    import msgpack
except ImportError:
    msgpack = None  # Will use json fallback

import json

logger = logging.getLogger(__name__)


# Constants
WAL_MAGIC = 0x57573443  # "WW4C" in hex
_WAL_VERSION = 2  # Version 2: HMAC-SHA256 integrity
WAL_VERSION = 1  # Keep for backward compat reference
SEGMENT_MAX_SIZE = 64 * 1024 * 1024  # 64MB per segment
HEADER_SIZE = 4 + 8 + 8 + 2 + 2  # magic + lsn + ts + op(H) + len(H) = 24 bytes
CHECKSUM_SIZE = 4  # v1 CRC32 size (backward compat)
HMAC_SIZE = 32  # v2 HMAC-SHA256 size


def _get_wal_key() -> bytes:
    """
    Get WAL HMAC key from environment.

    Returns:
        32-byte key from T4DM_WAL_KEY env var, or dev default
    """
    key = os.environ.get("T4DM_WAL_KEY", "")
    if key:
        return key.encode("utf-8")
    return b"t4dm-dev-wal-key-change-in-production"


def _compute_wal_hmac(data: bytes) -> bytes:
    """
    Compute HMAC-SHA256 for WAL entry data.

    Args:
        data: Entry header + payload bytes

    Returns:
        32-byte HMAC digest
    """
    return _hmac.new(_get_wal_key(), data, hashlib.sha256).digest()


class WALOperation(IntEnum):
    """Operations that get logged to WAL."""
    # Buffer operations
    BUFFER_ADD = 1
    BUFFER_REMOVE = 2
    BUFFER_PROMOTE = 3
    BUFFER_UPDATE = 4

    # Gate learning
    GATE_WEIGHT_UPDATE = 10
    GATE_BIAS_UPDATE = 11

    # Scorer learning
    SCORER_WEIGHT_UPDATE = 20
    SCORER_BIAS_UPDATE = 21

    # Neuromodulator state
    DOPAMINE_EXPECTATION = 30
    DOPAMINE_RPE = 31
    SEROTONIN_MOOD = 32
    NOREPINEPHRINE_AROUSAL = 33
    ACETYLCHOLINE_ENCODING = 34

    # Eligibility traces
    TRACE_CREATE = 40
    TRACE_UPDATE = 41
    TRACE_DECAY = 42
    TRACE_CLEAR = 43

    # Cluster index
    CLUSTER_ADD = 50
    CLUSTER_REMOVE = 51
    CLUSTER_UPDATE = 52

    # Consolidation
    CONSOLIDATION_START = 60
    CONSOLIDATION_COMMIT = 61
    CONSOLIDATION_ROLLBACK = 62

    # Checkpoint markers
    CHECKPOINT_START = 100
    CHECKPOINT_COMPLETE = 101

    # System
    SYSTEM_START = 200
    SYSTEM_SHUTDOWN = 201


class WALCorruptionError(Exception):
    """Raised when WAL data fails integrity check."""


@dataclass
class WALEntry:
    """A single WAL entry."""
    lsn: int  # Log Sequence Number (monotonic)
    timestamp_ns: int  # Nanoseconds since epoch
    operation: WALOperation
    payload: dict[str, Any]
    checksum: int = 0

    def serialize(self) -> bytes:
        """Serialize entry to bytes using HMAC-SHA256 (v2 format)."""
        if msgpack:
            payload_bytes = msgpack.packb(self.payload, use_bin_type=True)
        else:
            payload_bytes = json.dumps(self.payload).encode("utf-8")

        # Pack header
        header = struct.pack(
            ">IQQHH",  # big-endian: magic, lsn, timestamp, op, payload_len
            WAL_MAGIC,
            self.lsn,
            self.timestamp_ns,
            self.operation,
            len(payload_bytes),
        )

        # Compute HMAC-SHA256 over header + payload (v2)
        data = header + payload_bytes
        hmac_digest = _compute_wal_hmac(data)

        # Append HMAC (32 bytes)
        return data + hmac_digest

    @classmethod
    def deserialize(cls, data: bytes) -> tuple[WALEntry, int]:
        """
        Deserialize entry from bytes.
        Returns (entry, bytes_consumed).
        Raises WALCorruptionError on integrity failure.

        Supports both v1 (CRC32, 4 bytes) and v2 (HMAC, 32 bytes) formats.
        V1 entries are accepted with a warning for backward compatibility.
        """
        if len(data) < HEADER_SIZE + CHECKSUM_SIZE:
            raise WALCorruptionError(f"Data too short: {len(data)} bytes")

        # Unpack header
        magic, lsn, timestamp_ns, op, payload_len = struct.unpack(
            ">IQQHH", data[:HEADER_SIZE]
        )

        if magic != WAL_MAGIC:
            raise WALCorruptionError(f"Invalid magic: {hex(magic)}")

        # Detect format version by attempting to read both sizes
        # Try v2 (HMAC) first
        total_size_v2 = HEADER_SIZE + payload_len + HMAC_SIZE
        total_size_v1 = HEADER_SIZE + payload_len + CHECKSUM_SIZE

        # Determine which version based on available data
        if len(data) >= total_size_v2:
            # Attempt v2 (HMAC) verification
            payload_bytes = data[HEADER_SIZE:HEADER_SIZE + payload_len]
            stored_hmac = data[HEADER_SIZE + payload_len:total_size_v2]

            # Compute expected HMAC
            computed_hmac = _compute_wal_hmac(data[:HEADER_SIZE + payload_len])

            # Constant-time comparison
            if _hmac.compare_digest(computed_hmac, stored_hmac):
                # V2 entry verified
                if msgpack:
                    payload = msgpack.unpackb(payload_bytes, raw=False)
                else:
                    payload = json.loads(payload_bytes.decode("utf-8"))

                entry = cls(
                    lsn=lsn,
                    timestamp_ns=timestamp_ns,
                    operation=WALOperation(op),
                    payload=payload,
                    checksum=0,  # Not used in v2
                )
                return entry, total_size_v2

        # Try v1 (CRC32) fallback for backward compatibility
        if len(data) >= total_size_v1:
            payload_bytes = data[HEADER_SIZE:HEADER_SIZE + payload_len]
            stored_checksum = struct.unpack(
                ">I", data[HEADER_SIZE + payload_len:total_size_v1]
            )[0]

            # Verify CRC32
            computed_checksum = zlib.crc32(data[:HEADER_SIZE + payload_len]) & 0xFFFFFFFF
            if computed_checksum == stored_checksum:
                # V1 entry verified - log warning
                logger.warning(
                    f"WAL entry at LSN {lsn} uses legacy CRC32 format (v1). "
                    "Consider rewriting WAL with HMAC (v2) for better security."
                )

                if msgpack:
                    payload = msgpack.unpackb(payload_bytes, raw=False)
                else:
                    payload = json.loads(payload_bytes.decode("utf-8"))

                entry = cls(
                    lsn=lsn,
                    timestamp_ns=timestamp_ns,
                    operation=WALOperation(op),
                    payload=payload,
                    checksum=stored_checksum,
                )
                return entry, total_size_v1

        # Neither format verified - corruption detected
        raise WALCorruptionError(
            f"Integrity check failed at LSN {lsn}: "
            f"neither HMAC (v2) nor CRC32 (v1) verification succeeded"
        )


@dataclass
class WALConfig:
    """WAL configuration."""
    directory: Path
    segment_max_size: int = SEGMENT_MAX_SIZE
    sync_mode: str = "fsync"  # "fsync", "fdatasync", "none"
    max_segments: int = 100  # Keep last N segments
    compression: bool = False  # Compress old segments


class WriteAheadLog:
    """
    Write-Ahead Log implementation.

    Thread-safe and async-compatible.

    Usage:
        wal = WriteAheadLog(config)
        await wal.open()

        # Log operations
        lsn = await wal.append(WALOperation.BUFFER_ADD, {"memory_id": "123", ...})

        # Force sync (optional, append does fsync by default)
        await wal.sync()

        # On recovery, iterate entries
        async for entry in wal.iter_entries(from_lsn=last_checkpoint_lsn):
            replay_operation(entry)

        await wal.close()
    """

    def __init__(self, config: WALConfig):
        self.config = config
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()
        self._current_lsn = 0
        self._current_segment: int | None = None
        self._current_file: Any | None = None  # File handle
        self._current_size = 0
        self._closed = True
        self._checkpoint_lsn = 0  # LSN of last checkpoint

    @property
    def current_lsn(self) -> int:
        """Current (next) LSN to be assigned."""
        return self._current_lsn

    @property
    def checkpoint_lsn(self) -> int:
        """LSN of last completed checkpoint."""
        return self._checkpoint_lsn

    async def open(self) -> None:
        """Open WAL for writing, recovering state from existing segments."""
        async with self._lock:
            if not self._closed:
                return

            # Ensure directory exists
            self.config.directory.mkdir(parents=True, exist_ok=True)

            # Find existing segments and recover LSN
            segments = self._list_segments()

            if segments:
                # Recover from existing segments
                self._current_segment = segments[-1]
                self._current_lsn = await self._recover_lsn(segments)
                logger.info(
                    f"WAL recovered: segment={self._current_segment}, "
                    f"lsn={self._current_lsn}"
                )
            else:
                # Fresh start
                self._current_segment = 1
                self._current_lsn = 1
                logger.info("WAL initialized fresh")

            # Open current segment for append
            await self._open_segment(self._current_segment)
            self._closed = False

            # Log system start
            await self._append_internal(
                WALOperation.SYSTEM_START,
                {"version": WAL_VERSION, "recovered_lsn": self._current_lsn}
            )

    async def close(self) -> None:
        """Close WAL gracefully."""
        async with self._lock:
            if self._closed:
                return

            # Log shutdown
            try:
                await self._append_internal(
                    WALOperation.SYSTEM_SHUTDOWN,
                    {"final_lsn": self._current_lsn}
                )
            except Exception as e:
                logger.warning(f"Failed to log shutdown: {e}")

            # Sync and close
            if self._current_file:
                try:
                    self._current_file.flush()
                    if self.config.sync_mode != "none":
                        os.fsync(self._current_file.fileno())
                    self._current_file.close()
                except Exception as e:
                    logger.error(f"Error closing WAL: {e}")

            self._current_file = None
            self._closed = True
            logger.info(f"WAL closed at LSN {self._current_lsn}")

    async def append(
        self,
        operation: WALOperation,
        payload: dict[str, Any],
    ) -> int:
        """
        Append entry to WAL.

        Returns assigned LSN.
        Blocks until entry is durable (fsynced).
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("WAL is closed")
            return await self._append_internal(operation, payload)

    async def _append_internal(
        self,
        operation: WALOperation,
        payload: dict[str, Any],
    ) -> int:
        """Internal append without lock."""
        # Create entry
        entry = WALEntry(
            lsn=self._current_lsn,
            timestamp_ns=time.time_ns(),
            operation=operation,
            payload=payload,
        )

        # Serialize
        data = entry.serialize()

        # Check if we need to rotate
        if self._current_size + len(data) > self.config.segment_max_size:
            await self._rotate_segment()

        # Write
        self._current_file.write(data)
        self._current_size += len(data)

        # Sync based on mode
        if self.config.sync_mode == "fsync":
            self._current_file.flush()
            os.fsync(self._current_file.fileno())
        elif self.config.sync_mode == "fdatasync":
            self._current_file.flush()
            os.fdatasync(self._current_file.fileno())

        # Increment LSN
        assigned_lsn = self._current_lsn
        self._current_lsn += 1

        return assigned_lsn

    async def sync(self) -> None:
        """Force sync to disk."""
        async with self._lock:
            if self._current_file and not self._closed:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())

    async def mark_checkpoint(self, checkpoint_lsn: int) -> None:
        """
        Mark a checkpoint LSN. WAL entries before this can be truncated.
        """
        async with self._lock:
            self._checkpoint_lsn = checkpoint_lsn
            await self._append_internal(
                WALOperation.CHECKPOINT_COMPLETE,
                {"checkpoint_lsn": checkpoint_lsn}
            )
            logger.info(f"WAL checkpoint marked at LSN {checkpoint_lsn}")

    async def truncate_before(self, lsn: int) -> int:
        """
        Remove segments that only contain entries before given LSN.
        Returns number of segments removed.
        """
        async with self._lock:
            segments = self._list_segments()
            removed = 0

            for seg_num in segments[:-1]:  # Never remove current segment
                seg_path = self._segment_path(seg_num)

                # Check if all entries in segment are before LSN
                max_lsn_in_segment = await self._get_max_lsn_in_segment(seg_path)

                if max_lsn_in_segment < lsn:
                    seg_path.unlink()
                    removed += 1
                    logger.info(f"Removed WAL segment {seg_num} (max_lsn={max_lsn_in_segment})")

            return removed

    async def iter_entries(
        self,
        from_lsn: int = 0,
        to_lsn: int | None = None,
        operations: set[WALOperation] | None = None,
    ) -> AsyncIterator[WALEntry]:
        """
        Iterate WAL entries in LSN order.

        Args:
            from_lsn: Start from this LSN (inclusive)
            to_lsn: Stop at this LSN (exclusive), None = to end
            operations: Filter to these operation types, None = all
        """
        # Flush current file to ensure all written data is visible to readers
        if self._current_file and not self._closed:
            self._current_file.flush()

        segments = self._list_segments()

        for seg_num in segments:
            seg_path = self._segment_path(seg_num)

            if not seg_path.exists():
                continue

            async for entry in self._read_segment(seg_path):
                # Filter by LSN range
                if entry.lsn < from_lsn:
                    continue
                if to_lsn is not None and entry.lsn >= to_lsn:
                    return

                # Filter by operation type
                if operations and entry.operation not in operations:
                    continue

                yield entry

    async def iter_uncommitted(self, checkpoint_lsn: int) -> AsyncIterator[WALEntry]:
        """
        Iterate entries after the checkpoint (for recovery replay).
        """
        async for entry in self.iter_entries(from_lsn=checkpoint_lsn + 1):
            # Skip checkpoint markers themselves
            if entry.operation in (WALOperation.CHECKPOINT_START, WALOperation.CHECKPOINT_COMPLETE):
                continue
            yield entry

    def _list_segments(self) -> list[int]:
        """List segment numbers in order."""
        segments = []
        for f in self.config.directory.glob("segment_*.wal"):
            try:
                num = int(f.stem.split("_")[1])
                segments.append(num)
            except (IndexError, ValueError):
                continue
        return sorted(segments)

    def _segment_path(self, segment_num: int) -> Path:
        """Get path for segment number."""
        return self.config.directory / f"segment_{segment_num:08d}.wal"

    async def _open_segment(self, segment_num: int) -> None:
        """Open segment file for append."""
        path = self._segment_path(segment_num)

        # Close existing if any
        if self._current_file:
            self._current_file.close()

        # Open for append binary
        self._current_file = open(path, "ab")
        self._current_segment = segment_num
        self._current_size = path.stat().st_size if path.exists() else 0

    async def _rotate_segment(self) -> None:
        """Rotate to a new segment."""
        # Close current
        if self._current_file:
            self._current_file.flush()
            os.fsync(self._current_file.fileno())
            self._current_file.close()

        # Open next
        next_segment = self._current_segment + 1
        await self._open_segment(next_segment)

        logger.info(f"WAL rotated to segment {next_segment}")

        # Cleanup old segments if over limit
        await self._cleanup_old_segments()

    async def _cleanup_old_segments(self) -> None:
        """Remove old segments beyond max_segments."""
        segments = self._list_segments()

        while len(segments) > self.config.max_segments:
            oldest = segments.pop(0)

            # Only remove if before checkpoint
            max_lsn = await self._get_max_lsn_in_segment(self._segment_path(oldest))
            if max_lsn < self._checkpoint_lsn:
                self._segment_path(oldest).unlink()
                logger.info(f"Cleaned up old WAL segment {oldest}")
            else:
                break  # Can't remove segments after checkpoint

    async def _recover_lsn(self, segments: list[int]) -> int:
        """Recover the next LSN from existing segments."""
        max_lsn = 0

        # Read last segment to find max LSN
        if segments:
            last_segment = self._segment_path(segments[-1])
            async for entry in self._read_segment(last_segment):
                max_lsn = max(max_lsn, entry.lsn)

        return max_lsn + 1

    async def _get_max_lsn_in_segment(self, path: Path) -> int:
        """Get maximum LSN in a segment file."""
        max_lsn = 0
        async for entry in self._read_segment(path):
            max_lsn = max(max_lsn, entry.lsn)
        return max_lsn

    async def _read_segment(self, path: Path) -> AsyncIterator[WALEntry]:
        """Read all entries from a segment file."""
        if not path.exists():
            return

        with open(path, "rb") as f:
            data = f.read()

        offset = 0
        while offset < len(data):
            try:
                entry, consumed = WALEntry.deserialize(data[offset:])
                offset += consumed
                yield entry
            except WALCorruptionError as e:
                logger.warning(f"WAL corruption at offset {offset} in {path}: {e}")
                # Try to skip to next valid entry
                offset += 1
                while offset < len(data):
                    try:
                        # Look for magic number
                        if struct.unpack(">I", data[offset:offset+4])[0] == WAL_MAGIC:
                            break
                    except struct.error:
                        pass
                    offset += 1
