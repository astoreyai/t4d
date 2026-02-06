"""
Integrity verification for T4DM persistence layer.

Provides cryptographic checksum computation and verification to detect
silent data corruption in segment files and other persisted data.

Uses SHA-256 for strong integrity guarantees.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable


class IntegrityError(Exception):
    """Raised when data fails integrity verification.

    Attributes:
        path: Optional path to the corrupted file
        expected: Expected checksum (if available)
        actual: Actual computed checksum (if available)
    """

    def __init__(
        self,
        message: str,
        path: Path | str | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        super().__init__(message)
        self.path = Path(path) if path else None
        self.expected = expected
        self.actual = actual


def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum of data.

    Args:
        data: Raw bytes to compute checksum for

    Returns:
        Hexadecimal string representation of SHA-256 hash (64 characters)

    Example:
        >>> compute_checksum(b"hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected: str) -> bool:
    """Verify data matches expected SHA-256 checksum.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        data: Raw bytes to verify
        expected: Expected SHA-256 checksum as hex string

    Returns:
        True if checksum matches, False otherwise

    Example:
        >>> data = b"hello world"
        >>> checksum = compute_checksum(data)
        >>> verify_checksum(data, checksum)
        True
        >>> verify_checksum(b"corrupted", checksum)
        False
    """
    actual = compute_checksum(data)
    # Use hmac.compare_digest for constant-time comparison
    import hmac
    return hmac.compare_digest(actual, expected)


def compute_file_checksum(path: Path) -> str:
    """Compute SHA-256 checksum of a file.

    Reads file in chunks to handle large files efficiently.

    Args:
        path: Path to file

    Returns:
        Hexadecimal SHA-256 hash

    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_checksum(path: Path, expected: str) -> bool:
    """Verify file matches expected SHA-256 checksum.

    Args:
        path: Path to file
        expected: Expected SHA-256 checksum as hex string

    Returns:
        True if checksum matches, False otherwise

    Raises:
        FileNotFoundError: If file does not exist
    """
    actual = compute_file_checksum(path)
    import hmac
    return hmac.compare_digest(actual, expected)


@runtime_checkable
class Checksumable(Protocol):
    """Protocol for objects that support checksum computation."""

    def get_checksum_data(self) -> bytes:
        """Return bytes to be checksummed."""
        ...


class ChecksumMixin:
    """Mixin providing checksum computation and verification for segment writers.

    Classes using this mixin should implement get_checksum_data() or
    override compute_checksum() to provide data for checksumming.

    The mixin stores checksums in a .checksum file alongside the data file.

    Example:
        class SegmentWriter(ChecksumMixin):
            def __init__(self, path: Path):
                self._path = path
                self._data = b""

            def write(self, data: bytes) -> None:
                self._data = data
                with open(self._path, "wb") as f:
                    f.write(data)
                self.write_checksum(self._path)

            def get_checksum_data(self) -> bytes:
                return self._data
    """

    def compute_checksum(self) -> str:
        """Compute checksum for this object's data.

        Override this method or implement get_checksum_data() to provide
        the bytes to be checksummed.

        Returns:
            SHA-256 checksum as hex string

        Raises:
            NotImplementedError: If neither this method nor get_checksum_data is implemented
        """
        if hasattr(self, "get_checksum_data"):
            data = self.get_checksum_data()
            return compute_checksum(data)
        raise NotImplementedError(
            "Subclass must implement get_checksum_data() or override compute_checksum()"
        )

    def write_checksum(self, data_path: Path) -> Path:
        """Write checksum to a sidecar file.

        Creates a .checksum file alongside the data file containing
        the SHA-256 checksum.

        Args:
            data_path: Path to the data file

        Returns:
            Path to the checksum file
        """
        checksum = self.compute_checksum()
        checksum_path = data_path.with_suffix(data_path.suffix + ".checksum")
        checksum_path.write_text(checksum)
        return checksum_path

    def verify_checksum(self, data_path: Path) -> bool:
        """Verify data matches stored checksum.

        Reads checksum from sidecar file and compares against computed checksum.

        Args:
            data_path: Path to the data file

        Returns:
            True if checksum matches

        Raises:
            IntegrityError: If checksum does not match or checksum file is missing
        """
        checksum_path = data_path.with_suffix(data_path.suffix + ".checksum")

        if not checksum_path.exists():
            raise IntegrityError(
                f"Checksum file not found: {checksum_path}",
                path=data_path,
            )

        expected = checksum_path.read_text().strip()
        actual = self.compute_checksum()

        import hmac
        if not hmac.compare_digest(actual, expected):
            raise IntegrityError(
                f"Checksum mismatch for {data_path}",
                path=data_path,
                expected=expected,
                actual=actual,
            )

        return True

    @staticmethod
    def verify_file_integrity(data_path: Path) -> bool:
        """Verify a file's integrity using its sidecar checksum file.

        This is a static method that can be used without instantiating
        the mixin, useful for verification during recovery.

        Args:
            data_path: Path to the data file

        Returns:
            True if checksum matches

        Raises:
            IntegrityError: If checksum does not match or files are missing
            FileNotFoundError: If data file does not exist
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        checksum_path = data_path.with_suffix(data_path.suffix + ".checksum")

        if not checksum_path.exists():
            raise IntegrityError(
                f"Checksum file not found: {checksum_path}",
                path=data_path,
            )

        expected = checksum_path.read_text().strip()
        actual = compute_file_checksum(data_path)

        import hmac
        if not hmac.compare_digest(actual, expected):
            raise IntegrityError(
                f"Checksum mismatch for {data_path}",
                path=data_path,
                expected=expected,
                actual=actual,
            )

        return True

    @staticmethod
    def write_file_checksum(data_path: Path) -> Path:
        """Write checksum file for an existing data file.

        This is a static method that can be used to add checksums
        to existing files.

        Args:
            data_path: Path to the data file

        Returns:
            Path to the checksum file

        Raises:
            FileNotFoundError: If data file does not exist
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        checksum = compute_file_checksum(data_path)
        checksum_path = data_path.with_suffix(data_path.suffix + ".checksum")
        checksum_path.write_text(checksum)
        return checksum_path
