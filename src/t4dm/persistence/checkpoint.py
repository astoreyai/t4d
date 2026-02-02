"""
Checkpoint Manager for T4DM

Handles periodic snapshots of all in-memory state for fast recovery.

Checkpoint Contents:
===================
- Gate neural network weights and biases
- Scorer neural network weights and biases
- Cluster index
- Buffer contents (pending memories)
- Eligibility traces
- Neuromodulator state (DA expectations, 5-HT mood, NE arousal)
- WAL position (LSN at checkpoint time)

Checkpoint Strategy:
===================
1. Time-based: Every N minutes (default: 5)
2. Operation-based: Every M operations (default: 1000)
3. On-demand: Before shutdown, on explicit request

File Format:
===========
Atomic write using temp file + rename pattern:
1. Write to checkpoint.tmp
2. fsync
3. Rename to checkpoint_NNNNNNNN.bin (LSN in filename)
4. Remove old checkpoints (keep last K)
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import hmac
import json
import logging
import os
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


# Magic bytes for checkpoint file
CHECKPOINT_MAGIC = b"WWCP"
CHECKPOINT_VERSION = 2


def _get_checkpoint_key() -> bytes:
    """Get HMAC key for checkpoint signing."""
    key = os.environ.get("T4DM_CHECKPOINT_KEY", "")
    if key:
        return key.encode("utf-8")
    logger.warning(
        "T4DM_CHECKPOINT_KEY not set, using default development key. "
        "Set this environment variable in production!"
    )
    return b"t4dm-dev-checkpoint-key-change-in-production"


class _StateEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and other non-JSON types."""
    def default(self, obj):
        if HAS_NUMPY:
            if isinstance(obj, np.ndarray):
                return {
                    "__numpy__": True,
                    "data": obj.tolist(),
                    "dtype": str(obj.dtype)
                }
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        if isinstance(obj, bytes):
            return {
                "__bytes__": True,
                "data": base64.b64encode(obj).decode("ascii")
            }
        return super().default(obj)


def _state_decoder(obj):
    """JSON decoder hook for numpy arrays and bytes."""
    if "__numpy__" in obj:
        if not HAS_NUMPY:
            raise ValueError("Checkpoint contains numpy arrays but numpy is not installed")
        return np.array(obj["data"], dtype=obj["dtype"])
    if "__bytes__" in obj:
        return base64.b64decode(obj["data"])
    return obj


class CheckpointableComponent(Protocol):
    """Protocol for components that can be checkpointed."""

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return state dict for checkpointing."""
        ...

    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint dict."""
        ...


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    directory: Path
    interval_seconds: float = 300.0  # 5 minutes
    operation_threshold: int = 1000  # After N operations
    max_checkpoints: int = 5  # Keep last N checkpoints
    compression: bool = True  # gzip compression
    verify_on_load: bool = True  # Verify checksum on load


@dataclass
class Checkpoint:
    """
    A checkpoint snapshot.

    Contains all state needed to restore the system.
    """
    # Metadata
    lsn: int  # WAL LSN at checkpoint time
    timestamp: float  # Unix timestamp
    version: int = CHECKPOINT_VERSION

    # Component states
    gate_state: dict[str, Any] | None = None
    scorer_state: dict[str, Any] | None = None
    buffer_state: dict[str, Any] | None = None
    cluster_state: dict[str, Any] | None = None
    traces_state: dict[str, Any] | None = None
    neuromod_state: dict[str, Any] | None = None

    # Custom component states (extensible)
    custom_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Integrity
    checksum: str = ""

    def serialize(self, compress: bool = True) -> bytes:
        """Serialize checkpoint to bytes using JSON and HMAC-SHA256."""
        # Build state dict (excluding checksum)
        state = {
            "lsn": self.lsn,
            "timestamp": self.timestamp,
            "version": self.version,
            "gate_state": self.gate_state,
            "scorer_state": self.scorer_state,
            "buffer_state": self.buffer_state,
            "cluster_state": self.cluster_state,
            "traces_state": self.traces_state,
            "neuromod_state": self.neuromod_state,
            "custom_states": self.custom_states,
        }

        # JSON serialize state
        serialized = json.dumps(state, cls=_StateEncoder).encode("utf-8")

        # Compute HMAC-SHA256 signature over serialized data
        key = _get_checkpoint_key()
        signature = hmac.new(key, serialized, hashlib.sha256).hexdigest()

        # Build final format: magic + version + signature_len + signature + data_len + data
        header = (
            CHECKPOINT_MAGIC +
            struct.pack(">H", CHECKPOINT_VERSION) +
            struct.pack(">H", len(signature)) +
            signature.encode("ascii") +
            struct.pack(">Q", len(serialized))
        )

        data = header + serialized

        if compress:
            data = gzip.compress(data, compresslevel=6)

        return data

    @classmethod
    def deserialize(cls, data: bytes) -> Checkpoint:
        """Deserialize checkpoint from bytes using JSON and HMAC-SHA256 verification."""
        # Decompress if needed
        if data[:2] == b"\x1f\x8b":  # gzip magic
            data = gzip.decompress(data)

        # Parse header
        if data[:4] != CHECKPOINT_MAGIC:
            raise ValueError(f"Invalid checkpoint magic: {data[:4]}")

        offset = 4
        version = struct.unpack(">H", data[offset:offset+2])[0]
        offset += 2

        # Reject v1 (pickle-based) checkpoints
        if version == 1:
            logger.error(
                "Checkpoint version 1 (pickle-based) is no longer supported due to RCE vulnerability. "
                "Please re-create checkpoint."
            )
            raise ValueError(
                "Pickle-based checkpoints (v1) are no longer supported. Re-create checkpoint."
            )

        if version > CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {version}")

        signature_len = struct.unpack(">H", data[offset:offset+2])[0]
        offset += 2

        stored_signature = data[offset:offset+signature_len].decode("ascii")
        offset += signature_len

        data_len = struct.unpack(">Q", data[offset:offset+8])[0]
        offset += 8

        serialized = data[offset:offset+data_len]

        # Verify HMAC-SHA256 signature (always mandatory)
        key = _get_checkpoint_key()
        computed_signature = hmac.new(key, serialized, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(computed_signature, stored_signature):
            raise ValueError(
                f"Checkpoint HMAC signature mismatch. "
                f"Data may be corrupted or tampered with."
            )

        # Deserialize JSON state
        try:
            state = json.loads(serialized.decode("utf-8"), object_hook=_state_decoder)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode checkpoint JSON: {e}")

        return cls(
            lsn=state["lsn"],
            timestamp=state["timestamp"],
            version=state["version"],
            gate_state=state.get("gate_state"),
            scorer_state=state.get("scorer_state"),
            buffer_state=state.get("buffer_state"),
            cluster_state=state.get("cluster_state"),
            traces_state=state.get("traces_state"),
            neuromod_state=state.get("neuromod_state"),
            custom_states=state.get("custom_states", {}),
            checksum=stored_signature,
        )


class CheckpointManager:
    """
    Manages checkpoint creation, storage, and restoration.

    Thread-safe and async-compatible.

    Usage:
        manager = CheckpointManager(config)
        await manager.start()

        # Register components
        manager.register_component("gate", gate)
        manager.register_component("scorer", scorer)

        # Checkpoints happen automatically, or force one:
        await manager.create_checkpoint(current_lsn)

        # On recovery:
        checkpoint = await manager.load_latest_checkpoint()
        if checkpoint:
            manager.restore_all(checkpoint)

        await manager.stop()
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self._components: dict[str, CheckpointableComponent] = {}
        self._lock = asyncio.Lock()
        self._operation_count = 0
        self._last_checkpoint_time = 0.0
        self._last_checkpoint_lsn = 0
        self._running = False
        self._checkpoint_task: asyncio.Task | None = None
        self._get_current_lsn: Callable[[], int] | None = None

    def register_component(
        self,
        name: str,
        component: CheckpointableComponent,
    ) -> None:
        """Register a component for checkpointing."""
        self._components[name] = component
        logger.debug(f"Registered checkpoint component: {name}")

    def set_lsn_provider(self, provider: Callable[[], int]) -> None:
        """Set function to get current WAL LSN."""
        self._get_current_lsn = provider

    def record_operation(self) -> None:
        """
        Record an operation for threshold-based checkpointing.
        Call this after each WAL append.
        """
        self._operation_count += 1

    async def start(self) -> None:
        """Start automatic checkpointing."""
        if self._running:
            return

        self.config.directory.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._last_checkpoint_time = time.time()

        # Start background checkpoint task
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        logger.info("Checkpoint manager started")

    async def stop(self) -> None:
        """Stop automatic checkpointing."""
        self._running = False

        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        logger.info("Checkpoint manager stopped")

    async def _checkpoint_loop(self) -> None:
        """Background loop for automatic checkpointing."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second

                # Check if checkpoint needed
                time_elapsed = time.time() - self._last_checkpoint_time
                ops_since_last = self._operation_count

                should_checkpoint = (
                    time_elapsed >= self.config.interval_seconds or
                    ops_since_last >= self.config.operation_threshold
                )

                if should_checkpoint and self._get_current_lsn:
                    current_lsn = self._get_current_lsn()
                    if current_lsn > self._last_checkpoint_lsn:
                        await self.create_checkpoint(current_lsn)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Back off on error

    async def create_checkpoint(self, lsn: int) -> Checkpoint:
        """
        Create a checkpoint at the given LSN.

        This is the core checkpoint operation:
        1. Collect state from all registered components
        2. Serialize to bytes
        3. Write atomically to disk
        4. Clean up old checkpoints
        """
        async with self._lock:
            logger.info(f"Creating checkpoint at LSN {lsn}")
            start_time = time.time()

            # Collect component states
            checkpoint = Checkpoint(
                lsn=lsn,
                timestamp=time.time(),
            )

            for name, component in self._components.items():
                try:
                    state = component.get_checkpoint_state()

                    # Map to known fields or custom
                    if name == "gate":
                        checkpoint.gate_state = state
                    elif name == "scorer":
                        checkpoint.scorer_state = state
                    elif name == "buffer":
                        checkpoint.buffer_state = state
                    elif name == "cluster":
                        checkpoint.cluster_state = state
                    elif name == "traces":
                        checkpoint.traces_state = state
                    elif name == "neuromod":
                        checkpoint.neuromod_state = state
                    else:
                        checkpoint.custom_states[name] = state

                except Exception as e:
                    logger.error(f"Failed to checkpoint component {name}: {e}")
                    raise

            # Serialize
            data = checkpoint.serialize(compress=self.config.compression)

            # Write atomically
            checkpoint_path = self._checkpoint_path(lsn)
            await self._atomic_write(checkpoint_path, data)

            # Update tracking
            self._last_checkpoint_time = time.time()
            self._last_checkpoint_lsn = lsn
            self._operation_count = 0

            # Clean up old checkpoints
            await self._cleanup_old_checkpoints()

            elapsed = time.time() - start_time
            logger.info(
                f"Checkpoint created: LSN={lsn}, size={len(data)}, "
                f"time={elapsed:.2f}s"
            )

            return checkpoint

    async def load_latest_checkpoint(self) -> Checkpoint | None:
        """Load the most recent valid checkpoint."""
        checkpoints = self._list_checkpoints()

        if not checkpoints:
            logger.info("No checkpoints found")
            return None

        # Try checkpoints from newest to oldest
        for lsn in reversed(checkpoints):
            path = self._checkpoint_path(lsn)

            try:
                checkpoint = await self._load_checkpoint(path)
                logger.info(f"Loaded checkpoint: LSN={checkpoint.lsn}")
                return checkpoint

            except Exception as e:
                logger.warning(f"Failed to load checkpoint {lsn}: {e}")
                continue

        logger.error("All checkpoints failed to load")
        return None

    async def load_checkpoint_at_lsn(self, lsn: int) -> Checkpoint | None:
        """Load checkpoint at specific LSN."""
        path = self._checkpoint_path(lsn)

        if not path.exists():
            return None

        return await self._load_checkpoint(path)

    async def _load_checkpoint(self, path: Path) -> Checkpoint:
        """Load and verify a checkpoint file."""
        data = path.read_bytes()
        return Checkpoint.deserialize(data)

    def restore_all(self, checkpoint: Checkpoint) -> dict[str, bool]:
        """
        Restore all components from checkpoint.

        Returns dict of component_name -> success.
        """
        results = {}

        # Restore known components
        component_states = [
            ("gate", checkpoint.gate_state),
            ("scorer", checkpoint.scorer_state),
            ("buffer", checkpoint.buffer_state),
            ("cluster", checkpoint.cluster_state),
            ("traces", checkpoint.traces_state),
            ("neuromod", checkpoint.neuromod_state),
        ]

        for name, state in component_states:
            if state is not None and name in self._components:
                try:
                    self._components[name].restore_from_checkpoint(state)
                    results[name] = True
                    logger.debug(f"Restored component: {name}")
                except Exception as e:
                    logger.error(f"Failed to restore {name}: {e}")
                    results[name] = False

        # Restore custom components
        for name, state in checkpoint.custom_states.items():
            if name in self._components:
                try:
                    self._components[name].restore_from_checkpoint(state)
                    results[name] = True
                    logger.debug(f"Restored custom component: {name}")
                except Exception as e:
                    logger.error(f"Failed to restore custom {name}: {e}")
                    results[name] = False

        return results

    def _checkpoint_path(self, lsn: int) -> Path:
        """Get path for checkpoint at LSN."""
        ext = ".bin.gz" if self.config.compression else ".bin"
        return self.config.directory / f"checkpoint_{lsn:016d}{ext}"

    def _list_checkpoints(self) -> list[int]:
        """List checkpoint LSNs in order."""
        checkpoints = []

        for pattern in ["checkpoint_*.bin", "checkpoint_*.bin.gz"]:
            for path in self.config.directory.glob(pattern):
                try:
                    # Extract LSN from filename
                    stem = path.stem
                    stem = stem.removesuffix(".bin")  # Remove .bin from .bin.gz case
                    lsn = int(stem.split("_")[1])
                    checkpoints.append(lsn)
                except (IndexError, ValueError):
                    continue

        return sorted(set(checkpoints))

    async def _atomic_write(self, path: Path, data: bytes) -> None:
        """Write data atomically using temp file + rename."""
        # Write to temp file in same directory
        tmp_path = path.with_suffix(".tmp")

        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            tmp_path.rename(path)

            # Sync directory to ensure rename is durable
            dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        except Exception:
            # Clean up temp file on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    async def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = self._list_checkpoints()

        while len(checkpoints) > self.config.max_checkpoints:
            oldest_lsn = checkpoints.pop(0)
            path = self._checkpoint_path(oldest_lsn)

            # Also check for uncompressed version
            alt_path = path.with_suffix(".bin") if path.suffix == ".gz" else path.with_suffix(".bin.gz")

            for p in [path, alt_path]:
                if p.exists():
                    p.unlink()
                    logger.debug(f"Removed old checkpoint: {p}")

    @property
    def last_checkpoint_lsn(self) -> int:
        """LSN of last checkpoint."""
        return self._last_checkpoint_lsn


# Helper mixin for components
class CheckpointableMixin:
    """
    Mixin to make a class checkpointable.

    Subclasses should override _get_state_keys() to specify
    which attributes to checkpoint.
    """

    def _get_state_keys(self) -> list[str]:
        """Override to specify attributes to checkpoint."""
        return []

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {}
        for key in self._get_state_keys():
            if hasattr(self, key):
                value = getattr(self, key)
                # Handle PyTorch tensors/state_dicts
                if hasattr(value, "state_dict"):
                    state[key] = value.state_dict()
                elif hasattr(value, "cpu"):  # Tensor
                    state[key] = value.cpu().numpy()
                else:
                    state[key] = value
        return state

    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint dict."""
        for key, value in state.items():
            if hasattr(self, key):
                current = getattr(self, key)
                # Handle PyTorch state_dicts
                if hasattr(current, "load_state_dict") and isinstance(value, dict):
                    current.load_state_dict(value)
                else:
                    setattr(self, key, value)
