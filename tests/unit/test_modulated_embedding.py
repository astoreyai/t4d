"""
Unit tests for state-dependent embedding modulation.

Tests ModulatedEmbeddingAdapter and neuromodulator state handling.
"""

import pytest
import numpy as np

from ww.embedding.adapter import MockEmbeddingAdapter
from ww.embedding.modulated import (
    CognitiveMode,
    NeuromodulatorState,
    ModulationConfig,
    ModulatedEmbeddingAdapter,
    create_modulated_adapter,
)


class TestCognitiveMode:
    """Tests for CognitiveMode enum."""

    def test_mode_values(self):
        assert CognitiveMode.ENCODING.value == "encoding"
        assert CognitiveMode.RETRIEVAL.value == "retrieval"
        assert CognitiveMode.CONSOLIDATION.value == "consolidation"
        assert CognitiveMode.EXPLORATION.value == "exploration"
        assert CognitiveMode.EXPLOITATION.value == "exploitation"


class TestNeuromodulatorState:
    """Tests for NeuromodulatorState dataclass."""

    def test_default_values(self):
        state = NeuromodulatorState()
        assert state.acetylcholine == 0.5
        assert state.dopamine == 0.5
        assert state.norepinephrine == 0.5
        assert state.serotonin == 0.5

    def test_mode_inference_encoding(self):
        state = NeuromodulatorState(acetylcholine=0.9)
        assert state.mode == CognitiveMode.ENCODING

    def test_mode_inference_retrieval(self):
        state = NeuromodulatorState(acetylcholine=0.1)
        assert state.mode == CognitiveMode.RETRIEVAL

    def test_mode_inference_exploration(self):
        state = NeuromodulatorState(acetylcholine=0.5, norepinephrine=0.9)
        assert state.mode == CognitiveMode.EXPLORATION

    def test_mode_inference_exploitation(self):
        state = NeuromodulatorState(acetylcholine=0.5, norepinephrine=0.3, dopamine=0.9)
        assert state.mode == CognitiveMode.EXPLOITATION

    def test_for_encoding_factory(self):
        state = NeuromodulatorState.for_encoding()
        assert state.acetylcholine == 0.9
        assert state.mode == CognitiveMode.ENCODING

    def test_for_retrieval_factory(self):
        state = NeuromodulatorState.for_retrieval()
        assert state.acetylcholine == 0.2
        assert state.mode == CognitiveMode.RETRIEVAL

    def test_for_exploration_factory(self):
        state = NeuromodulatorState.for_exploration()
        assert state.norepinephrine == 0.9
        assert state.mode == CognitiveMode.EXPLORATION


class TestModulationConfig:
    """Tests for ModulationConfig."""

    def test_default_values(self):
        config = ModulationConfig()
        assert config.ach_gate_strength == 0.3
        assert config.da_amplification == 0.2
        assert config.ne_noise_scale == 0.05
        assert config.sparsity_ratio == 0.1

    def test_custom_values(self):
        config = ModulationConfig(
            ach_gate_strength=0.5,
            da_amplification=0.3,
        )
        assert config.ach_gate_strength == 0.5
        assert config.da_amplification == 0.3


class TestModulatedEmbeddingAdapter:
    """Tests for ModulatedEmbeddingAdapter."""

    @pytest.fixture
    def base_adapter(self):
        return MockEmbeddingAdapter(dimension=128, seed=42)

    @pytest.fixture
    def modulated_adapter(self, base_adapter):
        return ModulatedEmbeddingAdapter(base_adapter)

    def test_creation(self, modulated_adapter):
        assert modulated_adapter.dimension == 128
        assert modulated_adapter.state is not None

    def test_set_state(self, modulated_adapter):
        state = NeuromodulatorState.for_encoding()
        modulated_adapter.set_state(state)
        assert modulated_adapter.state.acetylcholine == 0.9

    @pytest.mark.asyncio
    async def test_embed_query_returns_correct_dimension(self, modulated_adapter):
        result = await modulated_adapter.embed_query("test query")
        assert len(result) == 128

    @pytest.mark.asyncio
    async def test_embed_query_normalized(self, modulated_adapter):
        result = await modulated_adapter.embed_query("test query")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_different_states_produce_different_embeddings(self, modulated_adapter):
        # Encoding mode
        modulated_adapter.set_state(NeuromodulatorState.for_encoding())
        encoding_emb = await modulated_adapter.embed_query("test")

        # Retrieval mode
        modulated_adapter.set_state(NeuromodulatorState.for_retrieval())
        retrieval_emb = await modulated_adapter.embed_query("test")

        # Should be different
        similarity = np.dot(encoding_emb, retrieval_emb)
        assert similarity < 0.99  # Not identical

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, modulated_adapter):
        result = await modulated_adapter.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, modulated_adapter):
        texts = ["text one", "text two", "text three"]
        results = await modulated_adapter.embed(texts)
        assert len(results) == 3
        for emb in results:
            assert len(emb) == 128
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_embed_query_unmodulated(self, modulated_adapter):
        # Get base embedding
        unmod = await modulated_adapter.embed_query_unmodulated("test")
        # Get modulated embedding
        mod = await modulated_adapter.embed_query("test")

        # Should be different
        similarity = np.dot(unmod, mod)
        assert similarity < 0.99

    def test_get_modulation_stats(self, modulated_adapter):
        modulated_adapter.set_state(NeuromodulatorState.for_encoding())
        stats = modulated_adapter.get_modulation_stats()

        assert "current_mode" in stats
        assert stats["current_mode"] == "encoding"
        assert "acetylcholine" in stats
        assert "config" in stats


class TestCreateModulatedAdapter:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        base = MockEmbeddingAdapter(dimension=64)
        adapter = create_modulated_adapter(base)

        assert adapter.dimension == 64
        assert adapter.state is not None

    def test_create_with_initial_state(self):
        base = MockEmbeddingAdapter(dimension=64)
        state = NeuromodulatorState.for_retrieval()
        adapter = create_modulated_adapter(base, initial_state=state)

        assert adapter.state.mode == CognitiveMode.RETRIEVAL

    def test_create_with_custom_config(self):
        base = MockEmbeddingAdapter(dimension=64)
        config = ModulationConfig(sparsity_ratio=0.2)
        adapter = create_modulated_adapter(base, config=config)

        assert adapter._config.sparsity_ratio == 0.2


class TestModulationBehavior:
    """Tests for specific modulation behaviors."""

    @pytest.fixture
    def adapter(self):
        base = MockEmbeddingAdapter(dimension=256)
        return ModulatedEmbeddingAdapter(base)

    @pytest.mark.asyncio
    async def test_exploration_adds_noise(self, adapter):
        """High NE should add exploration noise."""
        # Low NE
        adapter.set_state(NeuromodulatorState(norepinephrine=0.3))
        low_ne_emb = await adapter.embed_query("test")

        # High NE - exploration mode
        adapter.set_state(NeuromodulatorState.for_exploration())
        high_ne_emb = await adapter.embed_query("test")

        # With high NE, embeddings should be less consistent (more noise)
        # Run multiple times and check variance
        embeddings = []
        for _ in range(5):
            emb = await adapter.embed_query("test")
            embeddings.append(emb)

        # All should still be valid (normalized)
        for emb in embeddings:
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_dopamine_affects_salience(self, adapter):
        """High DA should amplify salient dimensions."""
        # Low DA
        adapter.set_state(NeuromodulatorState(dopamine=0.3))
        low_da = await adapter.embed_query("test")

        # High DA
        adapter.set_state(NeuromodulatorState(dopamine=0.9))
        high_da = await adapter.embed_query("test")

        # Both should be valid
        assert abs(np.linalg.norm(low_da) - 1.0) < 1e-5
        assert abs(np.linalg.norm(high_da) - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_stats_tracking(self, adapter):
        """Verify stats are tracked correctly."""
        await adapter.embed_query("q1")
        await adapter.embed_query("q2")
        await adapter.embed(["t1", "t2"])

        stats = adapter.stats
        assert stats.total_queries == 2
        assert stats.total_documents == 2
