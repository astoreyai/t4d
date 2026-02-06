"""Tests for persistence integrity checksums."""

import pytest
from pathlib import Path

from t4dm.persistence.integrity import (
    ChecksumMixin,
    IntegrityError,
    compute_checksum,
    compute_file_checksum,
    verify_checksum,
    verify_file_checksum,
)


class TestComputeChecksum:
    """Tests for compute_checksum function."""

    def test_empty_data(self):
        """Empty data produces known SHA-256 hash."""
        result = compute_checksum(b"")
        # SHA-256 of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_known_data(self):
        """Known data produces expected SHA-256 hash."""
        result = compute_checksum(b"hello world")
        assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_different_data_different_hash(self):
        """Different data produces different hashes."""
        hash1 = compute_checksum(b"data1")
        hash2 = compute_checksum(b"data2")
        assert hash1 != hash2

    def test_same_data_same_hash(self):
        """Same data produces identical hashes."""
        data = b"consistent data"
        hash1 = compute_checksum(data)
        hash2 = compute_checksum(data)
        assert hash1 == hash2

    def test_returns_hex_string(self):
        """Checksum is returned as 64-character hex string."""
        result = compute_checksum(b"test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_binary_data(self):
        """Handles arbitrary binary data."""
        data = bytes(range(256))
        result = compute_checksum(data)
        assert len(result) == 64


class TestVerifyChecksum:
    """Tests for verify_checksum function."""

    def test_valid_checksum(self):
        """Matching checksum returns True."""
        data = b"test data"
        checksum = compute_checksum(data)
        assert verify_checksum(data, checksum) is True

    def test_invalid_checksum(self):
        """Non-matching checksum returns False."""
        data = b"test data"
        wrong_checksum = "0" * 64
        assert verify_checksum(data, wrong_checksum) is False

    def test_corrupted_data(self):
        """Corrupted data fails verification."""
        original = b"original data"
        checksum = compute_checksum(original)
        corrupted = b"corrupted data"
        assert verify_checksum(corrupted, checksum) is False

    def test_single_bit_flip(self):
        """Single bit change is detected."""
        original = b"test data here"
        checksum = compute_checksum(original)
        # Flip one bit
        corrupted = bytes([original[0] ^ 1]) + original[1:]
        assert verify_checksum(corrupted, checksum) is False

    def test_case_sensitive(self):
        """Checksum comparison is case-sensitive (hex lowercase)."""
        data = b"test"
        checksum = compute_checksum(data)
        assert verify_checksum(data, checksum.upper()) is False


class TestComputeFileChecksum:
    """Tests for compute_file_checksum function."""

    def test_file_checksum(self, tmp_path):
        """Computes correct checksum for file."""
        data = b"file content"
        file_path = tmp_path / "test.dat"
        file_path.write_bytes(data)

        result = compute_file_checksum(file_path)
        expected = compute_checksum(data)
        assert result == expected

    def test_large_file(self, tmp_path):
        """Handles large files efficiently (chunked reading)."""
        # 1MB of data
        data = b"x" * (1024 * 1024)
        file_path = tmp_path / "large.dat"
        file_path.write_bytes(data)

        result = compute_file_checksum(file_path)
        expected = compute_checksum(data)
        assert result == expected

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            compute_file_checksum(tmp_path / "nonexistent.dat")

    def test_empty_file(self, tmp_path):
        """Handles empty files."""
        file_path = tmp_path / "empty.dat"
        file_path.write_bytes(b"")

        result = compute_file_checksum(file_path)
        expected = compute_checksum(b"")
        assert result == expected


class TestVerifyFileChecksum:
    """Tests for verify_file_checksum function."""

    def test_valid_file(self, tmp_path):
        """Valid file passes verification."""
        data = b"valid content"
        file_path = tmp_path / "valid.dat"
        file_path.write_bytes(data)
        checksum = compute_checksum(data)

        assert verify_file_checksum(file_path, checksum) is True

    def test_corrupted_file(self, tmp_path):
        """Corrupted file fails verification."""
        file_path = tmp_path / "corrupted.dat"
        file_path.write_bytes(b"original")
        checksum = compute_checksum(b"original")

        # Corrupt the file
        file_path.write_bytes(b"modified")

        assert verify_file_checksum(file_path, checksum) is False

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            verify_file_checksum(tmp_path / "missing.dat", "a" * 64)


class TestIntegrityError:
    """Tests for IntegrityError exception."""

    def test_basic_error(self):
        """Basic error message."""
        err = IntegrityError("Test error")
        assert str(err) == "Test error"
        assert err.path is None
        assert err.expected is None
        assert err.actual is None

    def test_error_with_path(self):
        """Error with file path."""
        err = IntegrityError("Corruption detected", path="/data/segment.dat")
        assert err.path == Path("/data/segment.dat")

    def test_error_with_checksums(self):
        """Error with expected and actual checksums."""
        err = IntegrityError(
            "Mismatch",
            expected="abc123",
            actual="def456",
        )
        assert err.expected == "abc123"
        assert err.actual == "def456"

    def test_full_error(self):
        """Error with all attributes."""
        err = IntegrityError(
            "Full error",
            path="/data/file.bin",
            expected="expected_hash",
            actual="actual_hash",
        )
        assert str(err) == "Full error"
        assert err.path == Path("/data/file.bin")
        assert err.expected == "expected_hash"
        assert err.actual == "actual_hash"


class SimpleChecksumWriter(ChecksumMixin):
    """Test implementation of ChecksumMixin."""

    def __init__(self):
        self._data = b""

    def set_data(self, data: bytes) -> None:
        self._data = data

    def get_checksum_data(self) -> bytes:
        return self._data


class TestChecksumMixin:
    """Tests for ChecksumMixin class."""

    def test_compute_checksum(self):
        """Computes checksum from get_checksum_data."""
        writer = SimpleChecksumWriter()
        writer.set_data(b"test data")

        result = writer.compute_checksum()
        expected = compute_checksum(b"test data")
        assert result == expected

    def test_write_checksum(self, tmp_path):
        """Writes checksum to sidecar file."""
        writer = SimpleChecksumWriter()
        writer.set_data(b"segment data")

        data_path = tmp_path / "segment.dat"
        data_path.write_bytes(b"segment data")

        checksum_path = writer.write_checksum(data_path)

        assert checksum_path == tmp_path / "segment.dat.checksum"
        assert checksum_path.exists()
        assert checksum_path.read_text() == compute_checksum(b"segment data")

    def test_verify_checksum_valid(self, tmp_path):
        """Verifies valid checksum passes."""
        writer = SimpleChecksumWriter()
        data = b"verified data"
        writer.set_data(data)

        data_path = tmp_path / "data.bin"
        data_path.write_bytes(data)
        writer.write_checksum(data_path)

        assert writer.verify_checksum(data_path) is True

    def test_verify_checksum_missing_file(self, tmp_path):
        """Raises IntegrityError when checksum file missing."""
        writer = SimpleChecksumWriter()
        writer.set_data(b"data")

        data_path = tmp_path / "no_checksum.bin"
        data_path.write_bytes(b"data")
        # No checksum file written

        with pytest.raises(IntegrityError) as exc_info:
            writer.verify_checksum(data_path)

        assert "Checksum file not found" in str(exc_info.value)
        assert exc_info.value.path == data_path

    def test_verify_checksum_mismatch(self, tmp_path):
        """Raises IntegrityError on checksum mismatch."""
        writer = SimpleChecksumWriter()
        writer.set_data(b"original")

        data_path = tmp_path / "mismatch.bin"
        data_path.write_bytes(b"original")
        writer.write_checksum(data_path)

        # Change the data (simulating corruption)
        writer.set_data(b"corrupted")

        with pytest.raises(IntegrityError) as exc_info:
            writer.verify_checksum(data_path)

        assert "Checksum mismatch" in str(exc_info.value)
        assert exc_info.value.path == data_path
        assert exc_info.value.expected is not None
        assert exc_info.value.actual is not None
        assert exc_info.value.expected != exc_info.value.actual

    def test_verify_file_integrity_static(self, tmp_path):
        """Static method verifies file integrity."""
        data = b"static verification"
        data_path = tmp_path / "static.bin"
        data_path.write_bytes(data)

        checksum_path = data_path.with_suffix(".bin.checksum")
        checksum_path.write_text(compute_checksum(data))

        assert ChecksumMixin.verify_file_integrity(data_path) is True

    def test_verify_file_integrity_corruption(self, tmp_path):
        """Static method detects file corruption."""
        data_path = tmp_path / "corrupt.bin"
        data_path.write_bytes(b"original content")

        checksum_path = data_path.with_suffix(".bin.checksum")
        checksum_path.write_text(compute_checksum(b"original content"))

        # Corrupt the file
        data_path.write_bytes(b"corrupted content")

        with pytest.raises(IntegrityError):
            ChecksumMixin.verify_file_integrity(data_path)

    def test_verify_file_integrity_missing_data(self, tmp_path):
        """Static method raises FileNotFoundError for missing data file."""
        with pytest.raises(FileNotFoundError):
            ChecksumMixin.verify_file_integrity(tmp_path / "missing.bin")

    def test_verify_file_integrity_missing_checksum(self, tmp_path):
        """Static method raises IntegrityError for missing checksum file."""
        data_path = tmp_path / "no_cksum.bin"
        data_path.write_bytes(b"data")

        with pytest.raises(IntegrityError):
            ChecksumMixin.verify_file_integrity(data_path)

    def test_write_file_checksum_static(self, tmp_path):
        """Static method writes checksum for existing file."""
        data = b"file to checksum"
        data_path = tmp_path / "tosum.bin"
        data_path.write_bytes(data)

        checksum_path = ChecksumMixin.write_file_checksum(data_path)

        assert checksum_path == tmp_path / "tosum.bin.checksum"
        assert checksum_path.read_text() == compute_checksum(data)

    def test_write_file_checksum_missing_file(self, tmp_path):
        """Static method raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ChecksumMixin.write_file_checksum(tmp_path / "missing.bin")

    def test_not_implemented_without_get_checksum_data(self):
        """Raises NotImplementedError if get_checksum_data not implemented."""

        class BadMixin(ChecksumMixin):
            pass

        writer = BadMixin()
        with pytest.raises(NotImplementedError):
            writer.compute_checksum()


class TestChecksumMixinWithSegmentFiles:
    """Integration tests simulating segment file usage."""

    def test_segment_write_read_cycle(self, tmp_path):
        """Full write-verify cycle for segment-like data."""

        class SegmentWriter(ChecksumMixin):
            def __init__(self, path: Path):
                self._path = path
                self._data: bytes = b""

            def write(self, data: bytes) -> None:
                self._data = data
                self._path.write_bytes(data)
                self.write_checksum(self._path)

            def get_checksum_data(self) -> bytes:
                return self._data

        segment_path = tmp_path / "segment_001.dat"
        writer = SegmentWriter(segment_path)

        # Write segment with checksum
        segment_data = b"vectors" * 1000 + b"metadata" * 100
        writer.write(segment_data)

        # Verify on read
        assert writer.verify_checksum(segment_path) is True

        # Verify with static method (simulating recovery)
        assert ChecksumMixin.verify_file_integrity(segment_path) is True

    def test_multiple_segments(self, tmp_path):
        """Multiple segments each with own checksum."""

        class SegmentWriter(ChecksumMixin):
            def __init__(self, data: bytes):
                self._data = data

            def get_checksum_data(self) -> bytes:
                return self._data

        segments = []
        for i in range(5):
            data = f"segment_{i}_content".encode() * 100
            path = tmp_path / f"segment_{i:03d}.dat"
            path.write_bytes(data)

            writer = SegmentWriter(data)
            writer.write_checksum(path)
            segments.append(path)

        # Verify all segments
        for path in segments:
            assert ChecksumMixin.verify_file_integrity(path) is True

        # Corrupt one segment
        segments[2].write_bytes(b"corrupted!")

        # Should fail for corrupted, pass for others
        for i, path in enumerate(segments):
            if i == 2:
                with pytest.raises(IntegrityError):
                    ChecksumMixin.verify_file_integrity(path)
            else:
                assert ChecksumMixin.verify_file_integrity(path) is True


class TestChecksumableProtocol:
    """Tests for Checksumable protocol."""

    def test_protocol_check(self):
        """Classes with get_checksum_data satisfy protocol."""
        from t4dm.persistence.integrity import Checksumable

        class ValidImpl:
            def get_checksum_data(self) -> bytes:
                return b"data"

        assert isinstance(ValidImpl(), Checksumable)

    def test_protocol_not_satisfied(self):
        """Classes without get_checksum_data don't satisfy protocol."""
        from t4dm.persistence.integrity import Checksumable

        class InvalidImpl:
            pass

        assert not isinstance(InvalidImpl(), Checksumable)
