"""
Cryptographic provenance for memories and learning events.

Provides HMAC-SHA256 signed provenance records to ensure memory integrity
and detect tampering. Every memory mutation should have a provenance record.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default key for development (override via T4DM_PROVENANCE_KEY env var)
_DEFAULT_KEY = b"t4dm-dev-provenance-key-change-in-production"


def _get_provenance_key() -> bytes:
    """Get provenance signing key from environment or default."""
    key_str = os.environ.get("T4DM_PROVENANCE_KEY", "")
    if key_str:
        return key_str.encode("utf-8")
    return _DEFAULT_KEY


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable provenance record for a memory or learning event."""

    content_hash: str        # SHA-256 of content
    origin: str              # e.g., "dg_processed", "api_store", "consolidation", "replay"
    created_at: float        # Unix timestamp (server-side, immutable)
    creator_id: str          # Module that created this (e.g., "hippocampus.ca3")
    signature: str           # HMAC-SHA256 signature
    record_id: str = field(default_factory=lambda: hashlib.sha256(
        os.urandom(32)
    ).hexdigest()[:16])


class IntegrityError(Exception):
    """Raised when provenance verification fails."""
    pass


def hash_content(content_bytes: bytes) -> str:
    """Compute SHA-256 hash of content bytes."""
    return hashlib.sha256(content_bytes).hexdigest()


def hash_embedding(embedding: np.ndarray) -> str:
    """Compute SHA-256 hash of an embedding vector."""
    return hashlib.sha256(embedding.tobytes()).hexdigest()


def _compute_signature(content_hash: str, origin: str, created_at: float, creator_id: str, key: bytes) -> str:
    """Compute HMAC-SHA256 signature over provenance fields."""
    message = f"{content_hash}|{origin}|{created_at}|{creator_id}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def sign_content(
    content_bytes: bytes,
    origin: str,
    creator_id: str,
    secret_key: Optional[bytes] = None,
) -> ProvenanceRecord:
    """Create a signed provenance record for content.

    Args:
        content_bytes: Raw content bytes to sign
        origin: Origin label (e.g., "api_store", "consolidation")
        creator_id: Module identifier (e.g., "hippocampus.ca3")
        secret_key: HMAC key (defaults to env var or dev key)

    Returns:
        Signed ProvenanceRecord
    """
    key = secret_key or _get_provenance_key()
    content_hash = hash_content(content_bytes)
    created_at = time.time()
    signature = _compute_signature(content_hash, origin, created_at, creator_id, key)

    return ProvenanceRecord(
        content_hash=content_hash,
        origin=origin,
        created_at=created_at,
        creator_id=creator_id,
        signature=signature,
    )


def sign_embedding(
    embedding: np.ndarray,
    origin: str,
    creator_id: str,
    secret_key: Optional[bytes] = None,
) -> ProvenanceRecord:
    """Create a signed provenance record for an embedding vector.

    Args:
        embedding: Numpy embedding vector
        origin: Origin label
        creator_id: Module identifier
        secret_key: HMAC key

    Returns:
        Signed ProvenanceRecord
    """
    return sign_content(embedding.tobytes(), origin, creator_id, secret_key)


def verify_provenance(
    record: ProvenanceRecord,
    content_bytes: bytes,
    secret_key: Optional[bytes] = None,
) -> bool:
    """Verify a provenance record against content.

    Args:
        record: ProvenanceRecord to verify
        content_bytes: Content bytes to verify against
        secret_key: HMAC key

    Returns:
        True if valid

    Raises:
        IntegrityError: If verification fails
    """
    key = secret_key or _get_provenance_key()

    # Verify content hash
    computed_hash = hash_content(content_bytes)
    if not hmac.compare_digest(computed_hash, record.content_hash):
        raise IntegrityError(
            f"Content hash mismatch: computed={computed_hash[:16]}..., "
            f"stored={record.content_hash[:16]}..."
        )

    # Verify HMAC signature
    expected_sig = _compute_signature(
        record.content_hash, record.origin, record.created_at, record.creator_id, key
    )
    if not hmac.compare_digest(expected_sig, record.signature):
        raise IntegrityError("Provenance signature verification failed")

    return True


def verify_embedding(
    record: ProvenanceRecord,
    embedding: np.ndarray,
    secret_key: Optional[bytes] = None,
) -> bool:
    """Verify provenance for an embedding vector.

    Args:
        record: ProvenanceRecord to verify
        embedding: Embedding to verify against
        secret_key: HMAC key

    Returns:
        True if valid

    Raises:
        IntegrityError: If verification fails
    """
    return verify_provenance(record, embedding.tobytes(), secret_key)
