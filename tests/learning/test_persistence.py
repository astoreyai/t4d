"""Tests for learning state persistence module."""

import gzip
import json
import pickle
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from ww.learning.persistence import (
    LearnedGateState,
    ScorerState,
    NeuromodulatorState,
    StatePersister,
)


class TestLearnedGateState:
    """Tests for LearnedGateState dataclass."""

    def test_creation(self):
        """Test basic creation."""
        state = LearnedGateState(
            weight_mean=np.array([0.1, 0.2, 0.3]),
            weight_covariance=np.array([1.0, 1.0, 1.0]),
            bias=0.5,
            use_diagonal=True,
            n_observations=100,
            decisions={"store": 50, "buffer": 30, "skip": 20},
        )

        assert state.bias == 0.5
        assert state.n_observations == 100
        assert state.use_diagonal is True
        assert isinstance(state.saved_at, datetime)
        assert state.version == "1.0.0"

    def test_to_dict(self):
        """Test conversion to dict."""
        state = LearnedGateState(
            weight_mean=np.array([0.1, 0.2, 0.3]),
            weight_covariance=np.array([1.0, 1.0, 1.0]),
            bias=0.5,
            use_diagonal=True,
            n_observations=100,
            decisions={"store": 50},
        )

        d = state.to_dict()

        assert d["weight_mean"] == [0.1, 0.2, 0.3]
        assert d["bias"] == 0.5
        assert d["use_diagonal"] is True
        assert "saved_at" in d

    def test_from_dict(self):
        """Test reconstruction from dict."""
        original = LearnedGateState(
            weight_mean=np.array([0.1, 0.2, 0.3]),
            weight_covariance=np.array([1.0, 1.0, 1.0]),
            bias=0.5,
            use_diagonal=True,
            n_observations=100,
            decisions={"store": 50},
        )

        d = original.to_dict()
        reconstructed = LearnedGateState.from_dict(d)

        assert np.allclose(reconstructed.weight_mean, original.weight_mean)
        assert reconstructed.bias == original.bias
        assert reconstructed.n_observations == original.n_observations

    def test_from_dict_missing_version(self):
        """Test from_dict with missing version defaults to 1.0.0."""
        d = {
            "weight_mean": [0.1, 0.2],
            "weight_covariance": [1.0, 1.0],
            "bias": 0.5,
            "use_diagonal": True,
            "n_observations": 10,
            "decisions": {},
            "saved_at": datetime.now().isoformat(),
        }

        state = LearnedGateState.from_dict(d)
        assert state.version == "1.0.0"


class TestScorerState:
    """Tests for ScorerState dataclass."""

    def test_creation(self):
        """Test basic creation."""
        state = ScorerState(
            layer_weights={"layer_0": np.array([[1, 2], [3, 4]])},
            layer_biases={"layer_0": np.array([0.1, 0.2])},
            n_training_steps=1000,
            recent_losses=[0.5, 0.4, 0.3],
        )

        assert state.n_training_steps == 1000
        assert len(state.recent_losses) == 3
        assert state.version == "1.0.0"


class TestNeuromodulatorState:
    """Tests for NeuromodulatorState dataclass."""

    def test_creation(self):
        """Test basic creation."""
        state = NeuromodulatorState(
            dopamine_expectations={"mem1": 0.7, "mem2": 0.3},
            serotonin_values={"mem1": 0.8},
            serotonin_mood=0.6,
            ne_reference_mean=np.array([0.0, 0.0]),
            ne_reference_std=np.array([1.0, 1.0]),
            ach_baseline_mode="encoding",
        )

        assert state.serotonin_mood == 0.6
        assert state.ach_baseline_mode == "encoding"
        assert len(state.dopamine_expectations) == 2


class TestStatePersister:
    """Tests for StatePersister class."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage path."""
        return tmp_path / "test_state"

    @pytest.fixture
    def persister(self, temp_storage):
        """Create persister with temp storage."""
        return StatePersister(storage_path=str(temp_storage))

    @pytest.fixture
    def persister_uncompressed(self, temp_storage):
        """Create uncompressed persister."""
        return StatePersister(storage_path=str(temp_storage), compress=False)

    def test_init_default(self, tmp_path):
        """Test default initialization creates storage dir."""
        with patch.object(Path, "home", return_value=tmp_path):
            persister = StatePersister()
            assert persister.storage_path.exists()
            assert persister.compress is True

    def test_init_custom_path(self, temp_storage):
        """Test initialization with custom path."""
        persister = StatePersister(storage_path=str(temp_storage))
        assert persister.storage_path == temp_storage
        assert temp_storage.exists()

    def test_get_path_compressed(self, persister):
        """Test path generation for compressed files."""
        path = persister._get_path("test_state")
        assert path.suffix == ".gz"
        assert ".json" in str(path)

    def test_get_path_uncompressed(self, persister_uncompressed):
        """Test path generation for uncompressed files."""
        path = persister_uncompressed._get_path("test_state")
        assert path.suffix == ".json"

    def test_save_load_json_compressed(self, persister):
        """Test saving and loading JSON compressed."""
        data = {"key": "value", "number": 42}
        path = persister._get_path("test")

        persister._save_json(path, data)
        loaded = persister._load_json(path)

        assert loaded == data

    def test_save_load_json_uncompressed(self, persister_uncompressed):
        """Test saving and loading JSON uncompressed."""
        data = {"key": "value", "number": 42}
        path = persister_uncompressed._get_path("test")

        persister_uncompressed._save_json(path, data)
        loaded = persister_uncompressed._load_json(path)

        assert loaded == data

    def test_load_json_not_found(self, persister):
        """Test loading non-existent file returns None."""
        path = persister._get_path("nonexistent")
        assert persister._load_json(path) is None

    def test_load_json_corrupted(self, persister):
        """Test loading corrupted file returns None."""
        path = persister._get_path("corrupted")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid data
        with gzip.open(path, "wt") as f:
            f.write("not valid json {{{")

        result = persister._load_json(path)
        assert result is None

    def test_load_json_bad_gzip(self, persister):
        """Test loading bad gzip file returns None."""
        path = persister._get_path("badgzip")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write non-gzip data to .gz file
        with open(path, "wb") as f:
            f.write(b"not gzipped data")

        result = persister._load_json(path)
        assert result is None


class TestStatePersisterGateOperations:
    """Tests for gate state operations."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage path."""
        return tmp_path / "test_state"

    @pytest.fixture
    def persister(self, temp_storage):
        """Create persister with temp storage."""
        return StatePersister(storage_path=str(temp_storage))

    @pytest.fixture
    def mock_gate(self):
        """Create a mock LearnedMemoryGate."""
        gate = MagicMock()
        gate.μ = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        gate.Σ = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        gate.b = 0.5
        gate.use_diagonal = True
        gate.n_observations = 100
        gate.decisions = {"store": 50, "buffer": 30, "skip": 20}
        gate.feature_dim = 3
        return gate

    def test_save_gate_state(self, persister, mock_gate):
        """Test saving gate state."""
        path = persister.save_gate_state(mock_gate)

        assert path.exists()
        assert "learned_gate" in path.name

    def test_load_gate_state(self, persister, mock_gate):
        """Test loading gate state."""
        persister.save_gate_state(mock_gate)
        state = persister.load_gate_state()

        assert state is not None
        assert np.allclose(state.weight_mean, mock_gate.μ)
        assert state.bias == mock_gate.b
        assert state.n_observations == mock_gate.n_observations

    def test_load_gate_state_not_found(self, persister):
        """Test loading when no state exists."""
        state = persister.load_gate_state()
        assert state is None

    def test_load_gate_state_invalid_data(self, persister):
        """Test loading invalid gate state returns None."""
        # Save invalid data
        path = persister._get_path("learned_gate")
        persister._save_json(path, {"invalid": "data"})

        state = persister.load_gate_state()
        assert state is None

    def test_restore_gate(self, persister, mock_gate):
        """Test restoring gate from saved state."""
        persister.save_gate_state(mock_gate)

        # Create new gate to restore into
        new_gate = MagicMock()
        new_gate.feature_dim = 3

        result = persister.restore_gate(new_gate)

        assert result is True
        assert np.allclose(new_gate.μ, mock_gate.μ)

    def test_restore_gate_not_found(self, persister):
        """Test restore returns False when no state."""
        gate = MagicMock()
        gate.feature_dim = 3

        result = persister.restore_gate(gate)
        assert result is False

    def test_restore_gate_dimension_mismatch(self, persister, mock_gate):
        """Test restore fails on dimension mismatch."""
        persister.save_gate_state(mock_gate)

        # Gate with different dimension
        new_gate = MagicMock()
        new_gate.feature_dim = 10  # Different from saved (3)

        result = persister.restore_gate(new_gate)
        assert result is False


class TestStatePersisterScorerOperations:
    """Tests for scorer state operations."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage path."""
        return tmp_path / "test_state"

    @pytest.fixture
    def persister(self, temp_storage):
        """Create persister with temp storage."""
        return StatePersister(storage_path=str(temp_storage))

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock LearnedRetrievalScorer."""
        scorer = MagicMock()

        # Mock layer with weight and bias
        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight.detach.return_value.cpu.return_value.numpy.return_value = np.array([[1, 2], [3, 4]])
        layer.bias = MagicMock()
        layer.bias.detach.return_value.cpu.return_value.numpy.return_value = np.array([0.1, 0.2])

        scorer.fusion_net = [layer]
        return scorer

    def test_save_scorer_state(self, persister, mock_scorer):
        """Test saving scorer state."""
        path = persister.save_scorer_state(mock_scorer, n_steps=100, losses=[0.5, 0.4])

        assert path.exists()
        assert "learned_scorer" in path.name

    def test_save_scorer_state_no_losses(self, persister, mock_scorer):
        """Test saving scorer state without losses."""
        path = persister.save_scorer_state(mock_scorer)

        assert path.exists()

    def test_load_scorer_state(self, persister, mock_scorer):
        """Test loading scorer state."""
        persister.save_scorer_state(mock_scorer, n_steps=100, losses=[0.5])
        state = persister.load_scorer_state()

        assert state is not None
        assert state.n_training_steps == 100
        assert state.recent_losses == [0.5]

    def test_load_scorer_state_not_found(self, persister):
        """Test loading when no scorer state exists."""
        state = persister.load_scorer_state()
        assert state is None

    def test_load_scorer_state_corrupted(self, persister):
        """Test loading corrupted scorer state returns None."""
        path = persister.storage_path / "learned_scorer.pkl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid pickle
        with gzip.open(path, "wb") as f:
            f.write(b"not valid pickle data")

        state = persister.load_scorer_state()
        assert state is None


class TestStatePersisterNeuromodulatorOperations:
    """Tests for neuromodulator state operations."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage path."""
        return tmp_path / "test_state"

    @pytest.fixture
    def persister(self, temp_storage):
        """Create persister with temp storage."""
        return StatePersister(storage_path=str(temp_storage))

    @pytest.fixture
    def mock_orchestra(self):
        """Create a mock NeuromodulatorOrchestra."""
        orchestra = MagicMock()

        # Dopamine
        orchestra.dopamine._expectations = {"mem1": 0.7, "mem2": 0.3}

        # Serotonin
        orchestra.serotonin._long_term_values = {"mem1": 0.8}
        orchestra.serotonin.get_current_mood.return_value = 0.6

        # Norepinephrine
        orchestra.norepinephrine._reference_mean = np.array([0.0, 0.0])
        orchestra.norepinephrine._reference_std = np.array([1.0, 1.0])

        # Acetylcholine
        orchestra.acetylcholine._baseline_mode = MagicMock()
        orchestra.acetylcholine._baseline_mode.value = "encoding"

        return orchestra

    def test_save_neuromodulator_state(self, persister, mock_orchestra):
        """Test saving neuromodulator state."""
        path = persister.save_neuromodulator_state(mock_orchestra)

        assert path.exists()
        assert "neuromodulators" in path.name

    def test_load_neuromodulator_state(self, persister, mock_orchestra):
        """Test loading neuromodulator state."""
        persister.save_neuromodulator_state(mock_orchestra)
        state = persister.load_neuromodulator_state()

        assert state is not None
        assert "mem1" in state.dopamine_expectations
        assert state.serotonin_mood == 0.6
        assert state.ach_baseline_mode == "encoding"

    def test_load_neuromodulator_state_not_found(self, persister):
        """Test loading when no neuromodulator state exists."""
        state = persister.load_neuromodulator_state()
        assert state is None

    def test_load_neuromodulator_state_corrupted(self, persister):
        """Test loading corrupted neuromodulator state returns None."""
        path = persister.storage_path / "neuromodulators.pkl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wb") as f:
            f.write(b"not valid pickle")

        state = persister.load_neuromodulator_state()
        assert state is None

    def test_save_neuromodulator_without_all_attrs(self, persister):
        """Test saving when orchestra doesn't have all attributes."""
        orchestra = MagicMock()

        # Minimal mocking - missing many attributes
        orchestra.dopamine = MagicMock(spec=[])  # No _expectations
        orchestra.serotonin = MagicMock(spec=["get_current_mood"])
        orchestra.serotonin.get_current_mood.return_value = 0.5
        orchestra.norepinephrine = MagicMock(spec=[])  # No reference
        orchestra.acetylcholine = MagicMock(spec=[])  # No baseline_mode

        path = persister.save_neuromodulator_state(orchestra)
        assert path.exists()


class TestStatePersisterBulkOperations:
    """Tests for bulk save operations."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage path."""
        return tmp_path / "test_state"

    @pytest.fixture
    def persister(self, temp_storage):
        """Create persister with temp storage."""
        return StatePersister(storage_path=str(temp_storage))

    @pytest.fixture
    def mock_gate(self):
        """Create a mock gate."""
        gate = MagicMock()
        gate.μ = np.array([0.1], dtype=np.float32)
        gate.Σ = np.array([1.0], dtype=np.float32)
        gate.b = 0.5
        gate.use_diagonal = True
        gate.n_observations = 10
        gate.decisions = {}
        return gate

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock scorer."""
        scorer = MagicMock()
        scorer.fusion_net = []
        return scorer

    @pytest.fixture
    def mock_orchestra(self):
        """Create a mock orchestra."""
        orchestra = MagicMock()
        orchestra.dopamine = MagicMock(spec=[])
        orchestra.serotonin = MagicMock(spec=["get_current_mood"])
        orchestra.serotonin.get_current_mood.return_value = 0.5
        orchestra.norepinephrine = MagicMock(spec=[])
        orchestra.acetylcholine = MagicMock(spec=[])
        return orchestra

    def test_save_all_with_all_components(self, persister, mock_gate, mock_scorer, mock_orchestra):
        """Test saving all components."""
        saved = persister.save_all(
            gate=mock_gate,
            scorer=mock_scorer,
            orchestra=mock_orchestra,
        )

        assert "gate" in saved
        assert "scorer" in saved
        assert "neuromodulators" in saved
        assert all(path.exists() for path in saved.values())

    def test_save_all_with_gate_only(self, persister, mock_gate):
        """Test saving only gate."""
        saved = persister.save_all(gate=mock_gate)

        assert "gate" in saved
        assert len(saved) == 1

    def test_save_all_with_no_components(self, persister):
        """Test saving with no components."""
        saved = persister.save_all()

        assert len(saved) == 0

    def test_get_storage_info(self, persister, mock_gate):
        """Test getting storage info."""
        persister.save_gate_state(mock_gate)

        info = persister.get_storage_info()

        assert "storage_path" in info
        assert "files" in info
        assert len(info["files"]) > 0
        assert "name" in info["files"][0]
        assert "size_bytes" in info["files"][0]
        assert "modified" in info["files"][0]

    def test_get_storage_info_empty(self, persister):
        """Test getting storage info with no files."""
        info = persister.get_storage_info()

        assert info["files"] == []
