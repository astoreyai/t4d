"""Tests for provenance signing and verification."""

import numpy as np
import pytest

from t4dm.core.provenance import (
    IntegrityError,
    ProvenanceRecord,
    hash_content,
    hash_embedding,
    sign_content,
    sign_embedding,
    verify_embedding,
    verify_provenance,
)


class TestSignVerifyRoundTrip:
    def test_sign_and_verify_content(self):
        content = b"test memory content"
        record = sign_content(content, origin="test", creator_id="test.module")
        assert verify_provenance(record, content)

    def test_sign_and_verify_embedding(self):
        emb = np.random.default_rng(42).random(128).astype(np.float32)
        record = sign_embedding(emb, origin="dg_processed", creator_id="hippocampus.dg")
        assert verify_embedding(record, emb)

    def test_tamper_detection_content(self):
        content = b"original content"
        record = sign_content(content, origin="test", creator_id="test")
        with pytest.raises(IntegrityError):
            verify_provenance(record, b"tampered content")

    def test_tamper_detection_embedding(self):
        emb = np.ones(128, dtype=np.float32)
        record = sign_embedding(emb, origin="test", creator_id="test")
        tampered = emb.copy()
        tampered[0] = 999.0
        with pytest.raises(IntegrityError):
            verify_embedding(record, tampered)

    def test_signature_tamper_detection(self):
        content = b"test"
        record = sign_content(content, origin="test", creator_id="test")
        # Create record with forged signature
        forged = ProvenanceRecord(
            content_hash=record.content_hash,
            origin=record.origin,
            created_at=record.created_at,
            creator_id=record.creator_id,
            signature="forged_signature_value",
        )
        with pytest.raises(IntegrityError):
            verify_provenance(forged, content)

    def test_custom_key(self):
        key = b"custom-test-key"
        content = b"secret content"
        record = sign_content(content, origin="test", creator_id="test", secret_key=key)
        assert verify_provenance(record, content, secret_key=key)
        # Wrong key should fail
        with pytest.raises(IntegrityError):
            verify_provenance(record, content, secret_key=b"wrong-key")

    def test_hash_deterministic(self):
        content = b"deterministic"
        h1 = hash_content(content)
        h2 = hash_content(content)
        assert h1 == h2

    def test_embedding_hash_deterministic(self):
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        h1 = hash_embedding(emb)
        h2 = hash_embedding(emb)
        assert h1 == h2

    def test_provenance_record_immutable(self):
        record = sign_content(b"test", origin="test", creator_id="test")
        with pytest.raises(AttributeError):
            record.content_hash = "tampered"
