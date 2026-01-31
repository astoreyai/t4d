"""
ATOM-P2-2: Tests for CA3 provenance tracking.

Ensures that CA3 can store patterns with optional provenance metadata
for tracking data lineage and origin.
"""

import numpy as np
import pytest


class TestCA3Provenance:
    """Test CA3 provenance storage."""

    def test_ca3_store_accepts_provenance(self):
        """P2-2: CA3 can store patterns with provenance."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig

        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128, ca3_dim=128))
        pattern = np.random.default_rng(42).random(128).astype(np.float32)

        # Create provenance record
        provenance = {
            "origin": "test",
            "creator_id": "test_suite",
            "timestamp": "2026-01-30T00:00:00Z",
            "version": "1.0.0",
        }

        # Access CA3 and store with provenance
        pattern_id = hpc.ca3.store(pattern, provenance=provenance)

        # Verify pattern was stored
        assert pattern_id is not None
        assert len(hpc.ca3._patterns) == 1

        # Verify provenance was recorded
        assert len(hpc.ca3._provenance_records) == 1
        assert 0 in hpc.ca3._provenance_records
        assert hpc.ca3._provenance_records[0] == provenance

    def test_ca3_store_without_provenance(self):
        """P2-2: CA3 can store patterns without provenance (backward compat)."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig

        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128, ca3_dim=128))
        pattern = np.random.default_rng(42).random(128).astype(np.float32)

        # Store without provenance (old behavior)
        pattern_id = hpc.ca3.store(pattern)

        # Verify pattern was stored
        assert pattern_id is not None
        assert len(hpc.ca3._patterns) == 1

        # Provenance should be empty
        assert len(hpc.ca3._provenance_records) == 0

    def test_ca3_provenance_eviction(self):
        """P2-2: Provenance is cleaned up when patterns are evicted."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig

        config = HippocampalConfig(ec_dim=128, ca3_dim=128, ca3_max_patterns=2)
        hpc = HippocampalCircuit(config)

        # Store 3 patterns with provenance (should evict first)
        for i in range(3):
            pattern = np.random.default_rng(i).random(128).astype(np.float32)
            provenance = {"index": i}
            hpc.ca3.store(pattern, provenance=provenance)

        # Should have only 2 patterns
        assert len(hpc.ca3._patterns) == 2

        # Provenance should only have records for remaining patterns
        assert len(hpc.ca3._provenance_records) == 2
        # First pattern (index 0) should be evicted
        assert 0 in hpc.ca3._provenance_records
        assert hpc.ca3._provenance_records[0]["index"] == 1
        assert 1 in hpc.ca3._provenance_records
        assert hpc.ca3._provenance_records[1]["index"] == 2
