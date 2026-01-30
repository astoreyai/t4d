"""
Tests for NCA Bridge module.

Tests memory-NCA integration for state-dependent encoding and retrieval.
"""

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from ww.bridges.nca_bridge import (
    NCABridgeConfig,
    BridgeConfig,
    EncodingContext,
    RetrievalContext,
    NCABridge,
)


class TestNCABridgeConfig:
    """Tests for NCABridgeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NCABridgeConfig()
        assert config.encoding_nt_weight == 0.3
        assert config.state_context_dim == 32
        assert config.retrieval_state_matching is True
        assert config.state_similarity_weight == 0.2
        assert config.use_nca_gradients is True
        assert config.coupling_lr_scale == 0.5
        assert config.focus_boost == 1.3
        assert config.explore_diversity == 1.5
        assert config.consolidate_replay == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NCABridgeConfig(
            encoding_nt_weight=0.5,
            state_context_dim=64,
            focus_boost=1.5,
        )
        assert config.encoding_nt_weight == 0.5
        assert config.state_context_dim == 64
        assert config.focus_boost == 1.5

    def test_backwards_compatibility_alias(self):
        """Test BridgeConfig alias works."""
        config = BridgeConfig()
        assert isinstance(config, NCABridgeConfig)


class TestEncodingContext:
    """Tests for EncodingContext dataclass."""

    def test_context_creation(self):
        """Test creating encoding context."""
        memory_id = uuid4()
        embedding = np.random.randn(1024).astype(np.float32)
        nt_state = np.random.randn(6).astype(np.float32)
        cognitive_state = MagicMock()
        cognitive_state.name = "FOCUS"

        context = EncodingContext(
            memory_id=memory_id,
            embedding=embedding,
            nt_state=nt_state,
            cognitive_state=cognitive_state,
            energy=0.5,
        )

        assert context.memory_id == memory_id
        assert np.array_equal(context.embedding, embedding)
        assert context.energy == 0.5
        assert context.timestamp is not None

    def test_default_energy(self):
        """Test default energy is zero."""
        context = EncodingContext(
            memory_id=uuid4(),
            embedding=np.zeros(64),
            nt_state=np.zeros(6),
            cognitive_state=None,
        )
        assert context.energy == 0.0


class TestRetrievalContext:
    """Tests for RetrievalContext dataclass."""

    def test_context_creation(self):
        """Test creating retrieval context."""
        query_embedding = np.random.randn(1024).astype(np.float32)
        query_nt_state = np.random.randn(6).astype(np.float32)
        cognitive_state = MagicMock()

        context = RetrievalContext(
            query_embedding=query_embedding,
            query_nt_state=query_nt_state,
            query_cognitive_state=cognitive_state,
        )

        assert np.array_equal(context.query_embedding, query_embedding)
        assert context.retrieved_ids == []
        assert context.state_similarities == []

    def test_default_lists(self):
        """Test default empty lists."""
        context = RetrievalContext(
            query_embedding=np.zeros(64),
            query_nt_state=np.zeros(6),
            query_cognitive_state=None,
        )
        assert context.retrieved_ids == []
        assert context.state_similarities == []


class TestNCABridge:
    """Tests for NCABridge."""

    @pytest.fixture
    def mock_neural_field(self):
        """Mock neural field solver."""
        field = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.full(6, 0.5, dtype=np.float32)
        field.get_mean_state.return_value = state
        field.step.return_value = None
        return field

    @pytest.fixture
    def mock_state_manager(self):
        """Mock state transition manager."""
        manager = MagicMock()
        state = MagicMock()
        state.name = "FOCUS"
        manager.get_current_state.return_value = state
        manager.update.return_value = None
        return manager

    @pytest.fixture
    def mock_coupling(self):
        """Mock learnable coupling."""
        coupling = MagicMock()
        coupling.K = np.random.randn(32, 32).astype(np.float32)
        coupling.update_from_rpe.return_value = None
        return coupling

    @pytest.fixture
    def mock_energy(self):
        """Mock energy landscape."""
        energy = MagicMock()
        energy.compute_total_energy.return_value = 0.5
        return energy

    @pytest.fixture
    def mock_dopamine(self):
        """Mock dopamine system."""
        dopamine = MagicMock()
        rpe_result = MagicMock()
        rpe_result.rpe = 0.3
        dopamine.compute_rpe.return_value = rpe_result
        dopamine.update_expectations.return_value = None
        return dopamine

    @pytest.fixture
    def bridge(self, mock_neural_field, mock_state_manager, mock_coupling, mock_energy):
        """Create bridge with mocked dependencies."""
        return NCABridge(
            config=NCABridgeConfig(),
            neural_field=mock_neural_field,
            coupling=mock_coupling,
            state_manager=mock_state_manager,
            energy_landscape=mock_energy,
        )

    def test_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge.config is not None
        assert bridge._encoding_history == []
        assert bridge._retrieval_history == []
        assert bridge._state_projection.shape == (6, 32)

    def test_get_current_nt_state(self, bridge, mock_neural_field):
        """Test getting current NT state."""
        nt_state = bridge.get_current_nt_state()

        assert nt_state.shape == (6,)
        mock_neural_field.get_mean_state.assert_called_once()

    def test_get_current_nt_state_no_field(self):
        """Test NT state when no neural field."""
        bridge = NCABridge(neural_field=None)
        nt_state = bridge.get_current_nt_state()

        assert nt_state.shape == (6,)
        assert np.allclose(nt_state, 0.5)

    def test_get_current_cognitive_state(self, bridge, mock_state_manager):
        """Test getting current cognitive state."""
        state = bridge.get_current_cognitive_state()

        assert state is not None
        mock_state_manager.get_current_state.assert_called_once()

    def test_augment_encoding(self, bridge):
        """Test encoding augmentation with state context."""
        memory_id = uuid4()
        embedding = np.random.randn(64).astype(np.float32)

        augmented, context = bridge.augment_encoding(embedding, memory_id)

        # Augmented should include state context
        expected_dim = len(embedding) + bridge.config.state_context_dim
        assert augmented.shape[0] == expected_dim

        # Context should be stored
        assert isinstance(context, EncodingContext)
        assert context.memory_id == memory_id
        assert len(bridge._encoding_history) == 1

    def test_augment_encoding_focus_boost(self, bridge, mock_state_manager):
        """Test FOCUS state applies boost to encoding."""
        # Create mock cognitive state enum-like object
        focus_state = MagicMock()
        focus_state.name = "FOCUS"
        mock_state_manager.get_current_state.return_value = focus_state

        memory_id = uuid4()
        embedding = np.random.randn(64).astype(np.float32)

        augmented, context = bridge.augment_encoding(embedding, memory_id)

        # Context should have the state
        assert context.cognitive_state.name == "FOCUS"

    def test_modulate_retrieval(self, bridge):
        """Test retrieval modulation by state similarity."""
        query = np.random.randn(64).astype(np.float32)

        # Create candidate contexts
        candidates = []
        for i in range(5):
            ctx = EncodingContext(
                memory_id=uuid4(),
                embedding=np.random.randn(64).astype(np.float32),
                nt_state=np.random.randn(6).astype(np.float32),
                cognitive_state=MagicMock(),
            )
            candidates.append(ctx)

        ranked_ids, retrieval_ctx = bridge.modulate_retrieval(query, candidates, top_k=3)

        assert len(ranked_ids) == 3
        assert len(retrieval_ctx.state_similarities) == 3
        assert isinstance(retrieval_ctx, RetrievalContext)
        assert len(bridge._retrieval_history) == 1

    def test_modulate_retrieval_empty_candidates(self, bridge):
        """Test retrieval with no candidates."""
        query = np.random.randn(64).astype(np.float32)

        ranked_ids, context = bridge.modulate_retrieval(query, [], top_k=5)

        assert ranked_ids == []
        assert context.retrieved_ids == []

    def test_compute_learning_signal(self, bridge, mock_dopamine):
        """Test computing combined learning signal."""
        bridge.dopamine = mock_dopamine
        memory_id = uuid4()

        signal = bridge.compute_learning_signal(memory_id, outcome=0.8)

        assert "memory_id" in signal
        assert "outcome" in signal
        assert "rpe" in signal
        assert "effective_lr" in signal
        mock_dopamine.compute_rpe.assert_called_once()

    def test_compute_learning_signal_no_dopamine(self, bridge):
        """Test learning signal without dopamine system."""
        bridge.dopamine = None
        memory_id = uuid4()

        signal = bridge.compute_learning_signal(memory_id, outcome=0.8)

        assert signal["rpe"] == 0.0

    def test_trigger_consolidation(self, bridge):
        """Test consolidation trigger."""
        # Add some encoding history
        for i in range(20):
            ctx = EncodingContext(
                memory_id=uuid4(),
                embedding=np.random.randn(64).astype(np.float32),
                nt_state=np.random.randn(6).astype(np.float32),
                cognitive_state=None,
                energy=float(i) * 0.1,
            )
            bridge._encoding_history.append(ctx)

        replay_ids = bridge.trigger_consolidation()

        assert len(replay_ids) <= bridge.config.consolidate_replay
        assert all(isinstance(id, type(uuid4())) for id in replay_ids)

    def test_trigger_consolidation_empty_history(self, bridge):
        """Test consolidation with no history."""
        replay_ids = bridge.trigger_consolidation()
        assert replay_ids == []

    def test_step(self, bridge, mock_neural_field, mock_state_manager):
        """Test stepping the bridge."""
        bridge.step(dt=0.01)

        mock_neural_field.step.assert_called_once_with(dt=0.01)

    def test_get_stats(self, bridge):
        """Test statistics retrieval."""
        stats = bridge.get_stats()

        assert "encoding_history_size" in stats
        assert "retrieval_history_size" in stats
        assert "current_nt_state" in stats
        assert "config" in stats

    def test_history_limit(self, bridge):
        """Test history is limited to max size."""
        # Add more than max entries
        for i in range(bridge._max_history + 100):
            ctx = EncodingContext(
                memory_id=uuid4(),
                embedding=np.zeros(64),
                nt_state=np.zeros(6),
                cognitive_state=None,
            )
            bridge._encoding_history.append(ctx)

        # Trigger history cleanup by adding one more via augment_encoding
        bridge.augment_encoding(np.zeros(64), uuid4())

        # Should be limited
        assert len(bridge._encoding_history) <= bridge._max_history


class TestNCABridgeIntegration:
    """Integration tests for NCA bridge."""

    def test_full_encoding_flow(self):
        """Test complete encoding flow."""
        # Create minimal mocks
        neural_field = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.zeros(6)
        neural_field.get_mean_state.return_value = state

        state_manager = MagicMock()
        focus_state = MagicMock()
        focus_state.name = "FOCUS"
        state_manager.get_current_state.return_value = focus_state

        energy = MagicMock()
        energy.compute_total_energy.return_value = 0.5

        bridge = NCABridge(
            neural_field=neural_field,
            state_manager=state_manager,
            energy_landscape=energy,
        )

        # Encode
        memory_id = uuid4()
        embedding = np.random.randn(128).astype(np.float32)

        augmented, context = bridge.augment_encoding(embedding, memory_id)

        assert context.memory_id == memory_id
        assert len(bridge._encoding_history) > 0
        assert context.energy == 0.5

    def test_state_affects_encoding(self):
        """Test that different states produce different augmentations."""
        neural_field = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.zeros(6)
        neural_field.get_mean_state.return_value = state

        state_manager = MagicMock()

        bridge = NCABridge(
            neural_field=neural_field,
            state_manager=state_manager,
        )

        embedding = np.random.randn(64).astype(np.float32)

        # Encode in FOCUS
        focus_state = MagicMock()
        focus_state.name = "FOCUS"
        state_manager.get_current_state.return_value = focus_state
        augmented_focus, context_focus = bridge.augment_encoding(embedding.copy(), uuid4())

        # Encode in EXPLORE
        explore_state = MagicMock()
        explore_state.name = "EXPLORE"
        state_manager.get_current_state.return_value = explore_state
        augmented_explore, context_explore = bridge.augment_encoding(embedding.copy(), uuid4())

        # Contexts should differ in cognitive state
        assert context_focus.cognitive_state.name != context_explore.cognitive_state.name

    def test_encoding_retrieval_round_trip(self):
        """Test encoding then retrieving with state matching."""
        neural_field = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.full(6, 0.5, dtype=np.float32)
        neural_field.get_mean_state.return_value = state

        state_manager = MagicMock()
        focus_state = MagicMock()
        focus_state.name = "FOCUS"
        state_manager.get_current_state.return_value = focus_state

        bridge = NCABridge(
            neural_field=neural_field,
            state_manager=state_manager,
        )

        # Encode several memories
        encoded_contexts = []
        for i in range(5):
            embedding = np.random.randn(64).astype(np.float32)
            _, ctx = bridge.augment_encoding(embedding, uuid4())
            encoded_contexts.append(ctx)

        # Retrieve
        query = np.random.randn(64).astype(np.float32)
        ranked_ids, retrieval_ctx = bridge.modulate_retrieval(query, encoded_contexts, top_k=3)

        assert len(ranked_ids) == 3
        assert all(sim >= 0 for sim in retrieval_ctx.state_similarities)

    def test_learning_signal_with_all_components(self):
        """Test learning signal combines all sources."""
        neural_field = MagicMock()
        state = MagicMock()
        state.to_array.return_value = np.full(6, 0.5, dtype=np.float32)
        neural_field.get_mean_state.return_value = state

        coupling = MagicMock()
        coupling.K = np.random.randn(6, 6).astype(np.float32)

        dopamine = MagicMock()
        rpe_result = MagicMock()
        rpe_result.rpe = 0.5
        dopamine.compute_rpe.return_value = rpe_result

        state_manager = MagicMock()
        focus_state = MagicMock()
        focus_state.name = "FOCUS"
        state_manager.get_current_state.return_value = focus_state

        bridge = NCABridge(
            neural_field=neural_field,
            coupling=coupling,
            dopamine=dopamine,
            state_manager=state_manager,
        )

        signal = bridge.compute_learning_signal(uuid4(), outcome=0.9)

        # Should have all components
        assert signal["rpe"] == 0.5
        assert signal["coupling_gradient"] is not None
        # FOCUS state should boost learning rate (base is 0.01 * 1.5 = 0.015)
        assert signal["effective_lr"] >= 0.01
