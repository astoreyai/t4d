"""
Tests for cold start priming system.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from t4dm.learning.persistence import (
    LearnedGateState,
    StatePersister,
)
from t4dm.learning.cold_start import (
    ContextSignals,
    ContextLoader,
    PopulationPrior,
    ColdStartManager,
)


class TestLearnedGateState:
    """Tests for LearnedGateState serialization."""

    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        state = LearnedGateState(
            weight_mean=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            weight_covariance=np.array([0.01, 0.02, 0.03], dtype=np.float32),
            bias=0.5,
            use_diagonal=True,
            n_observations=100,
            decisions={"store": 50, "buffer": 30, "skip": 20},
        )

        # Serialize and deserialize
        data = state.to_dict()
        restored = LearnedGateState.from_dict(data)

        assert np.allclose(restored.weight_mean, state.weight_mean)
        assert np.allclose(restored.weight_covariance, state.weight_covariance)
        assert restored.bias == state.bias
        assert restored.use_diagonal == state.use_diagonal
        assert restored.n_observations == state.n_observations
        assert restored.decisions == state.decisions


class TestStatePersister:
    """Tests for StatePersister."""

    def test_save_and_load_gate_state(self):
        """Test saving and loading gate state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StatePersister(storage_path=tmpdir, compress=False)

            # Create mock gate
            gate = MagicMock()
            gate.μ = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            gate.Σ = np.array([0.01, 0.02, 0.03], dtype=np.float32)
            gate.b = 0.5
            gate.use_diagonal = True
            gate.n_observations = 100
            gate.decisions = {"store": 50, "buffer": 30, "skip": 20}

            # Save
            path = persister.save_gate_state(gate)
            assert path.exists()

            # Load
            state = persister.load_gate_state()
            assert state is not None
            assert np.allclose(state.weight_mean, gate.μ)
            assert state.n_observations == 100

    def test_save_and_load_compressed(self):
        """Test compressed save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StatePersister(storage_path=tmpdir, compress=True)

            gate = MagicMock()
            gate.μ = np.zeros(100, dtype=np.float32)
            gate.Σ = np.ones(100, dtype=np.float32) * 0.1
            gate.b = 0.0
            gate.use_diagonal = True
            gate.n_observations = 50
            gate.decisions = {"store": 25, "buffer": 15, "skip": 10}

            path = persister.save_gate_state(gate)
            assert path.suffix == ".gz"

            state = persister.load_gate_state()
            assert state is not None
            assert len(state.weight_mean) == 100

    def test_load_nonexistent_returns_none(self):
        """Test loading when no state exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StatePersister(storage_path=tmpdir)
            state = persister.load_gate_state()
            assert state is None

    def test_restore_gate(self):
        """Test restoring gate from saved state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StatePersister(storage_path=tmpdir, compress=False)

            # Save state
            original_gate = MagicMock()
            original_gate.μ = np.array([0.5, 0.6, 0.7], dtype=np.float32)
            original_gate.Σ = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            original_gate.b = 0.2
            original_gate.use_diagonal = True
            original_gate.n_observations = 200
            original_gate.decisions = {"store": 100, "buffer": 50, "skip": 50}
            persister.save_gate_state(original_gate)

            # Create new gate to restore into
            new_gate = MagicMock()
            new_gate.feature_dim = 3  # Matches saved dimensions

            # Restore
            success = persister.restore_gate(new_gate)
            assert success

            # Check values were set
            assert np.allclose(new_gate.μ, original_gate.μ)
            assert np.allclose(new_gate.Σ, original_gate.Σ)
            assert new_gate.b == 0.2
            assert new_gate.n_observations == 200

    def test_restore_dimension_mismatch(self):
        """Test restore fails on dimension mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StatePersister(storage_path=tmpdir, compress=False)

            # Save 3-dim state
            gate = MagicMock()
            gate.μ = np.zeros(3, dtype=np.float32)
            gate.Σ = np.ones(3, dtype=np.float32)
            gate.b = 0.0
            gate.use_diagonal = True
            gate.n_observations = 10
            gate.decisions = {}
            persister.save_gate_state(gate)

            # Try to restore into gate with different dimension
            new_gate = MagicMock()
            new_gate.feature_dim = 100  # Different from saved

            success = persister.restore_gate(new_gate)
            assert not success  # Should fail due to mismatch


class TestContextSignals:
    """Tests for ContextSignals."""

    def test_to_feature_bias(self):
        """Test feature bias generation."""
        signals = ContextSignals(
            project_name="test_project",
            project_type="python",
            keywords=["memory", "learning"],
        )

        bias = signals.to_feature_bias(feature_dim=1000)
        assert len(bias) == 1000
        # Python projects get a small positive bias
        assert np.sum(np.abs(bias)) > 0


class TestContextLoader:
    """Tests for ContextLoader."""

    def test_detect_python_project(self):
        """Test Python project detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python project indicator
            (Path(tmpdir) / "pyproject.toml").touch()

            loader = ContextLoader(tmpdir)
            proj_type = loader._detect_project_type()
            assert proj_type == "python"

    def test_detect_typescript_project(self):
        """Test TypeScript project detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "tsconfig.json").touch()

            loader = ContextLoader(tmpdir)
            proj_type = loader._detect_project_type()
            assert proj_type == "typescript"

    def test_load_context_with_readme(self):
        """Test context loading with README."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create README
            readme = Path(tmpdir) / "README.md"
            readme.write_text("# Test Project\n\n## Features\n\n- Memory system\n")

            loader = ContextLoader(tmpdir)
            signals = loader.load_context(include_git=False)

            assert "Test Project" in signals.keywords
            assert "Features" in signals.keywords

    def test_load_context_with_claude_md(self):
        """Test context loading with CLAUDE.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text(
                "# My Project\n\n"
                "**Path**: `/home/user/project`\n\n"
                "IMPORTANT: Always test first\n"
            )

            loader = ContextLoader(tmpdir)
            signals = loader.load_context(
                claude_md_path=str(claude_md),
                include_git=False
            )

            assert len(signals.priority_patterns) > 0
            assert any("test" in p.lower() for p in signals.priority_patterns)


class TestPopulationPrior:
    """Tests for PopulationPrior."""

    def test_get_prior_weights(self):
        """Test prior weight generation."""
        prior = PopulationPrior()
        weights = prior.get_prior_weights(feature_dim=1143)

        assert len(weights) == 1143

        # Check neuromodulator section has expected structure
        neuro_start = 1088
        assert weights[neuro_start + 0] > 0  # DA RPE positive
        assert weights[neuro_start + 3] > 0  # ACh encoding positive
        assert weights[neuro_start + 5] < 0  # ACh retrieval negative


class TestColdStartManager:
    """Tests for ColdStartManager."""

    def test_initialize_with_no_persisted_state(self):
        """Test initialization without persisted state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock gate
            gate = MagicMock()
            gate.feature_dim = 1143
            gate.μ = np.zeros(1143, dtype=np.float32)
            gate.Σ = np.ones(1143, dtype=np.float32) * 0.1
            gate.n_observations = 0
            gate.cold_start_threshold = 100
            gate.decisions = {}

            persister = StatePersister(storage_path=tmpdir)
            manager = ColdStartManager(gate, persister=persister)

            # Initialize in empty temp dir (no persisted state)
            result = manager.initialize_session(
                working_dir=tmpdir,
                force_cold_start=True
            )

            assert result["strategy"] == "cold_start_priors"
            assert result["persisted_state_loaded"] is False
            assert result["context_loaded"] is True

    def test_initialize_with_persisted_state(self):
        """Test initialization with persisted state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StatePersister(storage_path=tmpdir, compress=False)

            # Save some state
            saved_gate = MagicMock()
            saved_gate.μ = np.random.randn(100).astype(np.float32)
            saved_gate.Σ = np.ones(100, dtype=np.float32) * 0.1
            saved_gate.b = 0.1
            saved_gate.use_diagonal = True
            saved_gate.n_observations = 500
            saved_gate.decisions = {"store": 300, "buffer": 100, "skip": 100}
            persister.save_gate_state(saved_gate)

            # Create new gate
            gate = MagicMock()
            gate.feature_dim = 100
            gate.n_observations = 0
            gate.cold_start_threshold = 100

            manager = ColdStartManager(gate, persister=persister)

            result = manager.initialize_session(working_dir=tmpdir)

            assert result["strategy"] == "persisted"
            assert result["persisted_state_loaded"] is True
            assert result["n_observations"] == 500

    def test_checkpoint(self):
        """Test checkpoint saves state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = MagicMock()
            gate.μ = np.zeros(10, dtype=np.float32)
            gate.Σ = np.ones(10, dtype=np.float32)
            gate.b = 0.0
            gate.use_diagonal = True
            gate.n_observations = 50
            gate.decisions = {"store": 30, "skip": 20}

            persister = StatePersister(storage_path=tmpdir)
            manager = ColdStartManager(gate, persister=persister)

            path = manager.checkpoint()
            assert path.exists()

    def test_finalize_session(self):
        """Test session finalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = MagicMock()
            gate.μ = np.zeros(10, dtype=np.float32)
            gate.Σ = np.ones(10, dtype=np.float32)
            gate.b = 0.0
            gate.use_diagonal = True
            gate.n_observations = 100
            gate.decisions = {"store": 60, "buffer": 20, "skip": 20}

            persister = StatePersister(storage_path=tmpdir)
            manager = ColdStartManager(gate, persister=persister)

            # Initialize first
            manager.initialize_session(working_dir=tmpdir, force_cold_start=True)

            # Finalize
            result = manager.finalize_session(session_outcome=0.8)

            assert result["saved_path"] is not None
            assert result["decisions"] == {"store": 60, "buffer": 20, "skip": 20}
            assert result["session_duration_minutes"] is not None

    def test_cold_start_progress(self):
        """Test cold start progress calculation."""
        gate = MagicMock()
        gate.n_observations = 50
        gate.cold_start_threshold = 100

        manager = ColdStartManager(gate)

        progress = manager.get_cold_start_progress()
        assert progress == 0.5

        # Test cap at 1.0
        gate.n_observations = 200
        progress = manager.get_cold_start_progress()
        assert progress == 1.0

    def test_blend_weights(self):
        """Test blend weight calculation."""
        gate = MagicMock()
        gate.n_observations = 25
        gate.cold_start_threshold = 100

        manager = ColdStartManager(gate)

        weights = manager.get_current_blend_weights()
        assert weights["heuristic"] == 0.75
        assert weights["learned"] == 0.25
