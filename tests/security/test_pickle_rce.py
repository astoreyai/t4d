"""Tests for pickle RCE elimination."""
import ast
import pytest
from pathlib import Path


class TestNoPickleInCheckpoint:
    """Verify pickle has been eliminated from checkpoint.py."""

    def test_pickle_not_imported(self):
        """Verify pickle is not imported in checkpoint.py."""
        source = Path("/mnt/projects/t4d/t4dm/src/ww/persistence/checkpoint.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "pickle", "pickle must not be imported"
            if isinstance(node, ast.ImportFrom):
                assert node.module != "pickle", "pickle must not be imported from"

    def test_checkpoint_roundtrip(self):
        """Verify checkpoint serialize/deserialize works without pickle."""
        from t4dm.persistence.checkpoint import Checkpoint
        cp = Checkpoint(lsn=42, timestamp=1234567890.0)
        cp.gate_state = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}
        cp.custom_states = {"test": {"key": "value"}}
        data = cp.serialize(compress=False)
        restored = Checkpoint.deserialize(data)
        assert restored.lsn == 42
        assert restored.gate_state["bias"] == 0.5
        assert restored.custom_states["test"]["key"] == "value"

    def test_checkpoint_roundtrip_compressed(self):
        """Verify checkpoint roundtrip with compression."""
        from t4dm.persistence.checkpoint import Checkpoint
        cp = Checkpoint(lsn=100, timestamp=9999999.0)
        cp.buffer_state = {"items": list(range(1000))}
        data = cp.serialize(compress=True)
        restored = Checkpoint.deserialize(data)
        assert restored.lsn == 100
        assert len(restored.buffer_state["items"]) == 1000

    def test_checkpoint_with_numpy_arrays(self):
        """Verify checkpoint handles numpy arrays correctly."""
        pytest.importorskip("numpy")
        import numpy as np
        from t4dm.persistence.checkpoint import Checkpoint

        cp = Checkpoint(lsn=1, timestamp=1000.0)
        cp.gate_state = {
            "weights": np.array([1.0, 2.0, 3.0]),
            "bias": np.float64(0.5)
        }
        data = cp.serialize(compress=False)
        restored = Checkpoint.deserialize(data)
        assert restored.lsn == 1
        assert isinstance(restored.gate_state["weights"], np.ndarray)
        assert np.allclose(restored.gate_state["weights"], [1.0, 2.0, 3.0])
        assert restored.gate_state["bias"] == 0.5

    def test_checkpoint_with_bytes(self):
        """Verify checkpoint handles bytes correctly."""
        from t4dm.persistence.checkpoint import Checkpoint

        cp = Checkpoint(lsn=5, timestamp=5000.0)
        cp.custom_states = {"binary": {"data": b"hello\x00world"}}
        data = cp.serialize(compress=False)
        restored = Checkpoint.deserialize(data)
        assert restored.custom_states["binary"]["data"] == b"hello\x00world"

    def test_checkpoint_tamper_detection(self):
        """Modify 1 byte of checkpoint, verify load raises."""
        from t4dm.persistence.checkpoint import Checkpoint
        cp = Checkpoint(lsn=1, timestamp=1000.0)
        data = cp.serialize(compress=False)
        # Tamper with a byte near the end (in the data portion)
        tampered = bytearray(data)
        tampered[-5] ^= 0xFF
        tampered = bytes(tampered)
        with pytest.raises(ValueError, match="HMAC signature mismatch"):
            Checkpoint.deserialize(tampered)

    def test_checkpoint_tamper_detection_compressed(self):
        """Verify tamper detection works with compression."""
        from t4dm.persistence.checkpoint import Checkpoint
        cp = Checkpoint(lsn=1, timestamp=1000.0)
        cp.gate_state = {"data": list(range(100))}
        data = cp.serialize(compress=True)
        # Tamper with middle of compressed data
        tampered = bytearray(data)
        tampered[len(tampered)//2] ^= 0xFF
        tampered = bytes(tampered)
        # Could fail on decompression or HMAC check
        with pytest.raises((ValueError, Exception)):
            Checkpoint.deserialize(tampered)

    def test_v1_checkpoint_rejected(self):
        """Verify v1 (pickle) checkpoints are rejected."""
        import struct
        import gzip
        import hashlib

        # Create a v1 checkpoint manually (without importing pickle)
        # We'll create a minimal valid v1 header that should be rejected
        state = b'{"lsn": 1, "timestamp": 1000.0}'  # Not actually pickle, but v1 format
        checksum = hashlib.sha256(state).hexdigest()
        header = (
            b"WWCP" +
            struct.pack(">H", 1) +  # version 1
            struct.pack(">H", len(checksum)) +
            checksum.encode("ascii") +
            struct.pack(">Q", len(state))
        )
        data = header + state

        from t4dm.persistence.checkpoint import Checkpoint
        with pytest.raises(ValueError, match="no longer supported"):
            Checkpoint.deserialize(data)

    def test_checkpoint_version_2(self):
        """Verify new checkpoints are version 2."""
        from t4dm.persistence.checkpoint import Checkpoint, CHECKPOINT_VERSION
        assert CHECKPOINT_VERSION == 2
        cp = Checkpoint(lsn=1, timestamp=1000.0)
        assert cp.version == 2
        data = cp.serialize(compress=False)
        restored = Checkpoint.deserialize(data)
        assert restored.version == 2

    def test_hmac_key_from_env(self, monkeypatch):
        """Verify HMAC key can be set from environment."""
        from t4dm.persistence.checkpoint import Checkpoint, _get_checkpoint_key

        # Set custom key
        monkeypatch.setenv("T4DM_CHECKPOINT_KEY", "test-key-12345")
        key = _get_checkpoint_key()
        assert key == b"test-key-12345"

        # Create checkpoint with custom key
        cp = Checkpoint(lsn=1, timestamp=1000.0)
        data = cp.serialize(compress=False)

        # Verify it can be restored with same key
        restored = Checkpoint.deserialize(data)
        assert restored.lsn == 1

        # Verify it fails with different key
        monkeypatch.setenv("T4DM_CHECKPOINT_KEY", "wrong-key")
        with pytest.raises(ValueError, match="HMAC signature mismatch"):
            Checkpoint.deserialize(data)

    def test_checkpoint_with_none_values(self):
        """Verify checkpoint handles None values correctly."""
        from t4dm.persistence.checkpoint import Checkpoint
        cp = Checkpoint(lsn=10, timestamp=10000.0)
        cp.gate_state = None
        cp.scorer_state = {"weights": None}
        data = cp.serialize(compress=False)
        restored = Checkpoint.deserialize(data)
        assert restored.gate_state is None
        assert restored.scorer_state["weights"] is None

    def test_checkpoint_with_nested_structures(self):
        """Verify checkpoint handles deeply nested data structures."""
        from t4dm.persistence.checkpoint import Checkpoint
        cp = Checkpoint(lsn=1, timestamp=1000.0)
        cp.custom_states = {
            "nested": {
                "level1": {
                    "level2": {
                        "level3": [1, 2, 3, {"key": "value"}]
                    }
                }
            }
        }
        data = cp.serialize(compress=False)
        restored = Checkpoint.deserialize(data)
        assert restored.custom_states["nested"]["level1"]["level2"]["level3"][3]["key"] == "value"
