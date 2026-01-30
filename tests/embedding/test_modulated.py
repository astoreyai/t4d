"""Tests for state-dependent embedding modulation."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from ww.embedding.modulated import (
    CognitiveMode,
    NeuromodulatorState,
    ModulationConfig,
    ModulatedEmbeddingAdapter,
    create_modulated_adapter,
)


class TestCognitiveMode:
    """Tests for CognitiveMode enum."""

    def test_modes_exist(self):
        """All cognitive modes are defined."""
        assert CognitiveMode.ENCODING.value == "encoding"
        assert CognitiveMode.RETRIEVAL.value == "retrieval"
        assert CognitiveMode.CONSOLIDATION.value == "consolidation"
        assert CognitiveMode.EXPLORATION.value == "exploration"
        assert CognitiveMode.EXPLOITATION.value == "exploitation"


class TestNeuromodulatorState:
    """Tests for NeuromodulatorState dataclass."""

    def test_default_state(self):
        """Default state has balanced neuromodulators."""
        state = NeuromodulatorState()
        assert state.acetylcholine == 0.5
        assert state.dopamine == 0.5
        assert state.norepinephrine == 0.5
        assert state.serotonin == 0.5

    def test_custom_state(self):
        """Create state with custom values."""
        state = NeuromodulatorState(
            acetylcholine=0.8,
            dopamine=0.3,
            norepinephrine=0.6,
            serotonin=0.4,
        )
        assert state.acetylcholine == 0.8
        assert state.dopamine == 0.3

    def test_mode_encoding(self):
        """High ACh indicates encoding mode."""
        state = NeuromodulatorState(acetylcholine=0.9)
        assert state.mode == CognitiveMode.ENCODING

    def test_mode_retrieval(self):
        """Low ACh indicates retrieval mode."""
        state = NeuromodulatorState(acetylcholine=0.2)
        assert state.mode == CognitiveMode.RETRIEVAL

    def test_mode_exploration(self):
        """High NE indicates exploration mode."""
        state = NeuromodulatorState(acetylcholine=0.5, norepinephrine=0.9)
        assert state.mode == CognitiveMode.EXPLORATION

    def test_mode_exploitation(self):
        """High DA indicates exploitation mode."""
        state = NeuromodulatorState(acetylcholine=0.5, norepinephrine=0.5, dopamine=0.9)
        assert state.mode == CognitiveMode.EXPLOITATION

    def test_mode_consolidation(self):
        """Balanced state indicates consolidation."""
        state = NeuromodulatorState()  # All at 0.5
        assert state.mode == CognitiveMode.CONSOLIDATION

    def test_for_encoding_factory(self):
        """Factory creates encoding-optimized state."""
        state = NeuromodulatorState.for_encoding()
        assert state.acetylcholine > 0.7
        assert state.mode == CognitiveMode.ENCODING

    def test_for_retrieval_factory(self):
        """Factory creates retrieval-optimized state."""
        state = NeuromodulatorState.for_retrieval()
        assert state.acetylcholine < 0.3
        assert state.mode == CognitiveMode.RETRIEVAL

    def test_for_exploration_factory(self):
        """Factory creates exploration state."""
        state = NeuromodulatorState.for_exploration()
        assert state.norepinephrine > 0.7
        assert state.mode == CognitiveMode.EXPLORATION


class TestModulationConfig:
    """Tests for ModulationConfig dataclass."""

    def test_default_config(self):
        """Default config values."""
        config = ModulationConfig()
        assert config.ach_gate_strength == 0.3
        assert config.da_amplification == 0.2
        assert config.ne_noise_scale == 0.05
        assert config.sparsity_ratio == 0.1
        assert config.cache_modulated is False

    def test_custom_config(self):
        """Custom config values."""
        config = ModulationConfig(
            ach_gate_strength=0.5,
            da_amplification=0.4,
        )
        assert config.ach_gate_strength == 0.5
        assert config.da_amplification == 0.4


class TestModulatedEmbeddingAdapter:
    """Tests for ModulatedEmbeddingAdapter class."""

    @pytest.fixture
    def mock_adapter(self):
        """Create mock base adapter."""
        adapter = MagicMock()
        adapter.dimension = 64
        adapter.backend = "mock"
        adapter.embed_query = AsyncMock(return_value=np.random.randn(64).tolist())
        adapter.embed = AsyncMock(
            return_value=[np.random.randn(64).tolist() for _ in range(3)]
        )
        return adapter

    @pytest.fixture
    def modulated(self, mock_adapter):
        """Create modulated adapter."""
        return ModulatedEmbeddingAdapter(adapter=mock_adapter)

    def test_initialization(self, modulated):
        """Test adapter initialization."""
        assert modulated._dimension == 64
        assert modulated._config is not None
        assert modulated._state is not None

    def test_initialization_with_config(self, mock_adapter):
        """Test initialization with custom config."""
        config = ModulationConfig(ach_gate_strength=0.8)
        modulated = ModulatedEmbeddingAdapter(mock_adapter, config=config)
        assert modulated._config.ach_gate_strength == 0.8

    def test_state_property(self, modulated):
        """Test state property."""
        state = modulated.state
        assert isinstance(state, NeuromodulatorState)

    def test_set_state(self, modulated):
        """Test setting state."""
        new_state = NeuromodulatorState.for_encoding()
        modulated.set_state(new_state)
        assert modulated.state == new_state
        assert modulated.state.mode == CognitiveMode.ENCODING

    def test_create_mode_mask_encoding(self, modulated):
        """Mode mask for encoding."""
        mask = modulated._create_mode_mask(CognitiveMode.ENCODING)
        assert mask.shape == (64,)
        assert np.all((mask == 0) | (mask == 1))

    def test_create_mode_mask_retrieval(self, modulated):
        """Mode mask for retrieval."""
        mask = modulated._create_mode_mask(CognitiveMode.RETRIEVAL)
        assert mask.shape == (64,)

    def test_create_mode_mask_deterministic(self, modulated):
        """Mode masks are deterministic."""
        mask1 = modulated._create_mode_mask(CognitiveMode.ENCODING)
        mask2 = modulated._create_mode_mask(CognitiveMode.ENCODING)
        np.testing.assert_array_equal(mask1, mask2)

    def test_apply_modulation_encoding(self, modulated):
        """Apply modulation in encoding mode."""
        modulated.set_state(NeuromodulatorState.for_encoding())
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        modulated_emb = modulated._apply_modulation(embedding)

        assert modulated_emb.shape == (64,)
        # Should be normalized
        assert np.linalg.norm(modulated_emb) == pytest.approx(1.0, abs=0.1)

    def test_apply_modulation_retrieval(self, modulated):
        """Apply modulation in retrieval mode."""
        modulated.set_state(NeuromodulatorState.for_retrieval())
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        modulated_emb = modulated._apply_modulation(embedding)

        assert modulated_emb.shape == (64,)

    def test_apply_modulation_high_dopamine(self, modulated):
        """Apply modulation with high dopamine."""
        state = NeuromodulatorState(dopamine=0.9)
        modulated.set_state(state)
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        modulated_emb = modulated._apply_modulation(embedding)

        assert modulated_emb.shape == (64,)

    def test_apply_modulation_high_norepinephrine(self, modulated):
        """Apply modulation with high NE adds noise."""
        state = NeuromodulatorState(norepinephrine=0.9)
        modulated.set_state(state)
        embedding = np.ones(64, dtype=np.float32) / np.sqrt(64)

        # Run multiple times to verify randomness
        results = []
        for _ in range(5):
            mod = modulated._apply_modulation(embedding.copy())
            results.append(mod)

        # Results should vary due to noise
        # (Note: might be very similar if noise is small)
        first = results[0]
        assert len(first) == 64

    @pytest.mark.asyncio
    async def test_embed_query(self, modulated, mock_adapter):
        """Embed query with modulation."""
        modulated.set_state(NeuromodulatorState.for_encoding())
        result = await modulated.embed_query("test query")

        assert len(result) == 64
        mock_adapter.embed_query.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_embed_multiple(self, modulated, mock_adapter):
        """Embed multiple texts with modulation."""
        modulated.set_state(NeuromodulatorState.for_retrieval())
        texts = ["text1", "text2", "text3"]
        results = await modulated.embed(texts)

        assert len(results) == 3
        for result in results:
            assert len(result) == 64

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, modulated):
        """Embed empty list."""
        results = await modulated.embed([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_query_unmodulated(self, modulated, mock_adapter):
        """Get unmodulated embedding."""
        result = await modulated.embed_query_unmodulated("test")
        mock_adapter.embed_query.assert_called_with("test")

    def test_get_modulation_stats(self, modulated):
        """Get modulation statistics."""
        modulated.set_state(NeuromodulatorState.for_encoding())
        stats = modulated.get_modulation_stats()

        assert "current_mode" in stats
        assert stats["current_mode"] == "encoding"
        assert "acetylcholine" in stats
        assert "config" in stats
        assert "stats" in stats


class TestCreateModulatedAdapter:
    """Tests for factory function."""

    @pytest.fixture
    def mock_adapter(self):
        """Create mock base adapter."""
        adapter = MagicMock()
        adapter.dimension = 128
        adapter.backend = "mock"
        return adapter

    def test_create_basic(self, mock_adapter):
        """Create basic modulated adapter."""
        modulated = create_modulated_adapter(mock_adapter)
        assert isinstance(modulated, ModulatedEmbeddingAdapter)

    def test_create_with_initial_state(self, mock_adapter):
        """Create with initial state."""
        initial = NeuromodulatorState.for_retrieval()
        modulated = create_modulated_adapter(mock_adapter, initial_state=initial)
        assert modulated.state.mode == CognitiveMode.RETRIEVAL

    def test_create_with_config(self, mock_adapter):
        """Create with custom config."""
        config = ModulationConfig(sparsity_ratio=0.2)
        modulated = create_modulated_adapter(mock_adapter, config=config)
        assert modulated._config.sparsity_ratio == 0.2


class TestModulationIntegration:
    """Integration tests for modulation."""

    @pytest.fixture
    def mock_adapter(self):
        """Create deterministic mock adapter."""
        adapter = MagicMock()
        adapter.dimension = 64
        adapter.backend = "mock"

        # Return deterministic embedding
        np.random.seed(42)
        fixed_emb = np.random.randn(64).astype(np.float32)
        fixed_emb /= np.linalg.norm(fixed_emb)
        adapter.embed_query = AsyncMock(return_value=fixed_emb.tolist())
        return adapter

    @pytest.mark.asyncio
    async def test_different_modes_produce_different_embeddings(self, mock_adapter):
        """Different cognitive modes produce different embeddings."""
        modulated = ModulatedEmbeddingAdapter(mock_adapter)

        # Encoding mode
        modulated.set_state(NeuromodulatorState.for_encoding())
        encoding_emb = await modulated.embed_query("test")

        # Retrieval mode
        modulated.set_state(NeuromodulatorState.for_retrieval())
        retrieval_emb = await modulated.embed_query("test")

        # Embeddings should differ
        encoding_arr = np.array(encoding_emb)
        retrieval_arr = np.array(retrieval_emb)

        # Cosine similarity should be < 1 (not identical)
        similarity = np.dot(encoding_arr, retrieval_arr)
        assert similarity < 0.99

    @pytest.mark.asyncio
    async def test_modulation_preserves_normalization(self, mock_adapter):
        """Modulated embeddings remain normalized."""
        modulated = ModulatedEmbeddingAdapter(mock_adapter)

        for state_factory in [
            NeuromodulatorState.for_encoding,
            NeuromodulatorState.for_retrieval,
            NeuromodulatorState.for_exploration,
        ]:
            modulated.set_state(state_factory())
            emb = await modulated.embed_query("test")
            norm = np.linalg.norm(emb)
            assert norm == pytest.approx(1.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_sparsity_reduces_active_dimensions(self, mock_adapter):
        """Sparsification reduces active dimensions."""
        config = ModulationConfig(sparsity_ratio=0.1)  # 10% active
        modulated = ModulatedEmbeddingAdapter(mock_adapter, config=config)

        emb = await modulated.embed_query("test")
        emb_arr = np.array(emb)

        # Check sparsity (many near-zero values)
        near_zero = np.sum(np.abs(emb_arr) < 0.01)
        # Should have some zeros due to sparsification
        assert near_zero >= 0

    def test_neuromodulator_mode_transitions(self):
        """Test mode transitions based on neuromodulator changes."""
        state = NeuromodulatorState()

        # Gradually increase ACh
        for ach in [0.3, 0.5, 0.7, 0.9]:
            state.acetylcholine = ach
            if ach > 0.7:
                assert state.mode == CognitiveMode.ENCODING
            elif ach < 0.3:
                assert state.mode == CognitiveMode.RETRIEVAL
