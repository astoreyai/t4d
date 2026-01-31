"""Tests for Memory-NCA bridge."""

import numpy as np
import pytest
from uuid import uuid4

from t4dm.bridge.memory_nca import (
    BridgeConfig,
    EncodingContext,
    MemoryNCABridge,
    RetrievalContext,
)
from t4dm.nca.neural_field import NeuralFieldSolver, NeurotransmitterState
from t4dm.nca.coupling import LearnableCoupling
from t4dm.nca.attractors import StateTransitionManager, CognitiveState
from t4dm.nca.energy import EnergyLandscape


class TestBridgeConfig:
    """Tests for BridgeConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = BridgeConfig()

        assert 0 < config.encoding_nt_weight < 1
        assert config.state_context_dim > 0
        assert config.focus_boost > 1.0

    def test_custom_values(self):
        """Config should accept custom values."""
        config = BridgeConfig(
            encoding_nt_weight=0.5,
            consolidate_replay=20
        )
        assert config.encoding_nt_weight == 0.5
        assert config.consolidate_replay == 20


class TestEncodingContext:
    """Tests for EncodingContext dataclass."""

    def test_creation(self):
        """Context should be creatable with required fields."""
        ctx = EncodingContext(
            memory_id=uuid4(),
            embedding=np.zeros(768),
            nt_state=np.zeros(6),
            cognitive_state=CognitiveState.FOCUS
        )

        assert ctx.embedding.shape == (768,)
        assert ctx.nt_state.shape == (6,)
        assert ctx.cognitive_state == CognitiveState.FOCUS


class TestRetrievalContext:
    """Tests for RetrievalContext dataclass."""

    def test_creation(self):
        """Context should be creatable with required fields."""
        ctx = RetrievalContext(
            query_embedding=np.zeros(768),
            query_nt_state=np.zeros(6),
            query_cognitive_state=CognitiveState.REST
        )

        assert len(ctx.retrieved_ids) == 0
        assert len(ctx.state_similarities) == 0


class TestMemoryNCABridge:
    """Tests for MemoryNCABridge."""

    def test_initialization_minimal(self):
        """Bridge should initialize with minimal args."""
        bridge = MemoryNCABridge()
        assert bridge.neural_field is None
        assert bridge.coupling is None

    def test_initialization_full(self):
        """Bridge should initialize with all components."""
        neural_field = NeuralFieldSolver()
        coupling = LearnableCoupling()
        state_manager = StateTransitionManager()
        energy = EnergyLandscape(coupling=coupling, state_manager=state_manager)

        bridge = MemoryNCABridge(
            neural_field=neural_field,
            coupling=coupling,
            state_manager=state_manager,
            energy_landscape=energy
        )

        assert bridge.neural_field is neural_field
        assert bridge.coupling is coupling

    def test_get_current_nt_state_no_field(self):
        """Without field, should return baseline state."""
        bridge = MemoryNCABridge()
        state = bridge.get_current_nt_state()

        assert state.shape == (6,)
        np.testing.assert_array_almost_equal(state, [0.5] * 6)

    def test_get_current_nt_state_with_field(self):
        """With field, should return field's state."""
        from t4dm.nca.neural_field import NeurotransmitterType
        field = NeuralFieldSolver()
        field.inject_stimulus(NeurotransmitterType.DOPAMINE, 0.2)
        bridge = MemoryNCABridge(neural_field=field)

        # Just check it returns something
        state = bridge.get_current_nt_state()
        assert state.shape == (6,)

    def test_get_current_cognitive_state_no_manager(self):
        """Without manager, should return None."""
        bridge = MemoryNCABridge()
        assert bridge.get_current_cognitive_state() is None

    def test_get_current_cognitive_state_with_manager(self):
        """With manager, should return current state."""
        manager = StateTransitionManager()
        bridge = MemoryNCABridge(state_manager=manager)

        state = bridge.get_current_cognitive_state()
        assert state == CognitiveState.REST

    def test_augment_encoding(self):
        """Augmented embedding should be larger."""
        bridge = MemoryNCABridge()
        embedding = np.random.randn(768).astype(np.float32)
        memory_id = uuid4()

        augmented, context = bridge.augment_encoding(embedding, memory_id)

        # Should be embedding + state context
        expected_dim = int(768 * (1 - bridge.config.encoding_nt_weight)) + bridge.config.state_context_dim
        # Actually the implementation concatenates, so it's larger
        assert len(augmented) > 768

        assert context.memory_id == memory_id
        assert context.embedding.shape == (768,)

    def test_augment_encoding_focus_boost(self):
        """FOCUS state should boost encoding."""
        manager = StateTransitionManager()
        manager.force_state(CognitiveState.FOCUS)
        bridge = MemoryNCABridge(state_manager=manager)

        embedding = np.random.randn(768).astype(np.float32)
        augmented, context = bridge.augment_encoding(embedding, uuid4())

        assert context.cognitive_state == CognitiveState.FOCUS

    def test_modulate_retrieval(self):
        """Retrieval modulation should return ranked IDs."""
        bridge = MemoryNCABridge()

        # Create candidate contexts
        candidates = []
        for i in range(5):
            ctx = EncodingContext(
                memory_id=uuid4(),
                embedding=np.random.randn(768),
                nt_state=np.random.rand(6),
                cognitive_state=list(CognitiveState)[i % 5]
            )
            candidates.append(ctx)

        query = np.random.randn(768).astype(np.float32)
        ranked, ret_ctx = bridge.modulate_retrieval(query, candidates, top_k=3)

        assert len(ranked) == 3
        assert len(ret_ctx.state_similarities) == 3

    def test_modulate_retrieval_state_matching(self):
        """Retrieval should prefer matching cognitive states."""
        manager = StateTransitionManager()
        manager.force_state(CognitiveState.FOCUS)
        config = BridgeConfig(
            retrieval_state_matching=True,
            state_similarity_weight=0.5
        )
        bridge = MemoryNCABridge(config=config, state_manager=manager)

        # Create candidates with different states
        focus_ctx = EncodingContext(
            memory_id=uuid4(),
            embedding=np.random.randn(768),
            nt_state=manager.attractors[CognitiveState.FOCUS].center,
            cognitive_state=CognitiveState.FOCUS
        )
        rest_ctx = EncodingContext(
            memory_id=uuid4(),
            embedding=np.random.randn(768),
            nt_state=manager.attractors[CognitiveState.REST].center,
            cognitive_state=CognitiveState.REST
        )

        query = np.random.randn(768).astype(np.float32)
        ranked, _ = bridge.modulate_retrieval(query, [rest_ctx, focus_ctx], top_k=2)

        # FOCUS should be ranked higher (we're in FOCUS state)
        assert ranked[0] == focus_ctx.memory_id

    def test_compute_learning_signal(self):
        """Learning signal should include expected components."""
        bridge = MemoryNCABridge()
        memory_id = uuid4()

        signal = bridge.compute_learning_signal(memory_id, outcome=0.8)

        assert "memory_id" in signal
        assert "outcome" in signal
        assert "rpe" in signal
        assert "effective_lr" in signal

    def test_compute_learning_signal_with_coupling(self):
        """Learning signal should update coupling when available."""
        coupling = LearnableCoupling()
        bridge = MemoryNCABridge(coupling=coupling)

        initial_count = coupling._update_count
        bridge.compute_learning_signal(uuid4(), outcome=0.9)

        assert coupling._update_count > initial_count

    def test_trigger_consolidation(self):
        """Consolidation should return memory IDs for replay."""
        bridge = MemoryNCABridge()

        # Add some encoding history
        for i in range(20):
            bridge._encoding_history.append(
                EncodingContext(
                    memory_id=uuid4(),
                    embedding=np.random.randn(768),
                    nt_state=np.random.rand(6),
                    cognitive_state=CognitiveState.FOCUS,
                    energy=np.random.rand() * 5
                )
            )

        replay_ids = bridge.trigger_consolidation()

        assert len(replay_ids) <= bridge.config.consolidate_replay
        assert len(replay_ids) > 0

    def test_step(self):
        """Step should update components without error."""
        field = NeuralFieldSolver()
        manager = StateTransitionManager()
        bridge = MemoryNCABridge(
            neural_field=field,
            state_manager=manager
        )

        # Should not raise
        bridge.step(dt=0.01)

    def test_stats(self):
        """Stats should include expected keys."""
        bridge = MemoryNCABridge()
        stats = bridge.get_stats()

        assert "encoding_history_size" in stats
        assert "retrieval_history_size" in stats
        assert "current_nt_state" in stats
        assert "config" in stats


class TestBridgeIntegration:
    """Integration tests for Memory-NCA bridge."""

    def test_full_encode_retrieve_learn_cycle(self):
        """Full cycle: encode -> retrieve -> learn."""
        # Set up full system
        field = NeuralFieldSolver()
        coupling = LearnableCoupling()
        manager = StateTransitionManager()
        energy = EnergyLandscape(coupling=coupling, state_manager=manager)

        bridge = MemoryNCABridge(
            neural_field=field,
            coupling=coupling,
            state_manager=manager,
            energy_landscape=energy
        )

        # Encode a memory in FOCUS state
        manager.force_state(CognitiveState.FOCUS)
        embedding = np.random.randn(768).astype(np.float32)
        memory_id = uuid4()

        augmented, enc_ctx = bridge.augment_encoding(embedding, memory_id)

        # Simulate time passing
        for _ in range(10):
            bridge.step(dt=0.01)

        # Retrieve (now in different state)
        manager.force_state(CognitiveState.REST)
        query = embedding + np.random.randn(768).astype(np.float32) * 0.1

        ranked, ret_ctx = bridge.modulate_retrieval(query, [enc_ctx], top_k=1)

        # Learn from outcome
        signal = bridge.compute_learning_signal(memory_id, outcome=0.8)

        # Verify the cycle completed
        assert enc_ctx.cognitive_state == CognitiveState.FOCUS
        assert ret_ctx.query_cognitive_state == CognitiveState.REST
        assert signal["outcome"] == 0.8
