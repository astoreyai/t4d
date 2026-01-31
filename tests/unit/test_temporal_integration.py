"""
Unit tests for temporal integration module.

Tests consolidation-plasticity integration components.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from uuid import uuid4

from t4dm.embedding.modulated import CognitiveMode, NeuromodulatorState
from t4dm.embedding.adapter import MockEmbeddingAdapter
from t4dm.embedding.modulated import ModulatedEmbeddingAdapter
from t4dm.temporal.integration import (
    adapt_orchestra_state,
    LearnedSalienceProvider,
    get_consolidation_state,
    get_sleep_replay_state,
    get_pattern_separation_state,
    PlasticityConfig,
    PlasticityCoordinator,
    create_plasticity_coordinator,
)


# Mock for OrchestraState
@dataclass
class MockOrchestraState:
    """Mock NeuromodulatorOrchestra state for testing."""
    dopamine_rpe: float = 0.0
    norepinephrine_gain: float = 1.0
    acetylcholine_mode: str = "balanced"
    serotonin_mood: float = 0.5
    inhibition_sparsity: float = 0.1


class TestAdaptOrchestraState:
    """Tests for orchestra state adaptation."""

    def test_balanced_mode(self):
        orchestra = MockOrchestraState(acetylcholine_mode="balanced")
        adapted = adapt_orchestra_state(orchestra)

        assert adapted.acetylcholine == 0.5

    def test_encoding_mode(self):
        orchestra = MockOrchestraState(acetylcholine_mode="encoding")
        adapted = adapt_orchestra_state(orchestra)

        assert adapted.acetylcholine == 0.9

    def test_retrieval_mode(self):
        orchestra = MockOrchestraState(acetylcholine_mode="retrieval")
        adapted = adapt_orchestra_state(orchestra)

        assert adapted.acetylcholine == 0.2

    def test_positive_rpe(self):
        orchestra = MockOrchestraState(dopamine_rpe=0.3)
        adapted = adapt_orchestra_state(orchestra)

        assert adapted.dopamine == 0.8  # 0.5 + 0.3

    def test_negative_rpe(self):
        orchestra = MockOrchestraState(dopamine_rpe=-0.2)
        adapted = adapt_orchestra_state(orchestra)

        assert adapted.dopamine == 0.3  # 0.5 - 0.2

    def test_high_gain(self):
        orchestra = MockOrchestraState(norepinephrine_gain=2.0)
        adapted = adapt_orchestra_state(orchestra)

        assert adapted.norepinephrine == 1.0

    def test_serotonin_preserved(self):
        orchestra = MockOrchestraState(serotonin_mood=0.8)
        adapted = adapt_orchestra_state(orchestra)

        assert adapted.serotonin == 0.8


class TestLearnedSalienceProvider:
    """Tests for learned salience provider."""

    def test_default_weights(self):
        provider = LearnedSalienceProvider()
        weights = provider.get_salience_weights(128)

        assert len(weights) == 128
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_set_gate_weights(self):
        provider = LearnedSalienceProvider()

        # Create mock gate weights (output_dim x input_dim)
        gate_weights = np.random.randn(64, 128).astype(np.float32)
        provider.set_gate_weights(gate_weights)

        weights = provider.get_salience_weights(128)

        assert len(weights) == 128
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_fallback_weights(self):
        provider = LearnedSalienceProvider()
        provider.fallback_weights = np.ones(64, dtype=np.float32) / 64

        weights = provider.get_salience_weights(64)

        assert len(weights) == 64

    def test_dimension_mismatch_uses_default(self):
        provider = LearnedSalienceProvider()
        provider.set_gate_weights(np.random.randn(64, 128))

        # Request different dimension
        weights = provider.get_salience_weights(256)

        # Should get uniform weights
        assert len(weights) == 256


class TestConsolidationStates:
    """Tests for predefined consolidation states."""

    def test_consolidation_state(self):
        state = get_consolidation_state()

        assert state.acetylcholine == 0.1  # Low
        assert state.dopamine == 0.7       # High
        assert state.norepinephrine == 0.2 # Low
        assert state.serotonin == 0.8      # High

    def test_sleep_replay_state(self):
        state = get_sleep_replay_state()

        assert state.acetylcholine == 0.05  # Very low
        assert state.norepinephrine == 0.1  # Very low
        assert state.serotonin == 0.9       # High

    def test_pattern_separation_state(self):
        state = get_pattern_separation_state()

        assert state.acetylcholine == 0.95  # High
        assert state.dopamine == 0.3        # Low
        assert state.norepinephrine == 0.5  # Moderate

    def test_consolidation_is_valid_neuromodulator_state(self):
        state = get_consolidation_state()

        assert isinstance(state, NeuromodulatorState)
        assert state.mode == CognitiveMode.RETRIEVAL  # Low ACh

    def test_pattern_separation_is_encoding_mode(self):
        state = get_pattern_separation_state()

        assert state.mode == CognitiveMode.ENCODING  # High ACh


class TestPlasticityConfig:
    """Tests for plasticity configuration."""

    def test_default_values(self):
        config = PlasticityConfig()

        assert config.max_update_per_outcome == 10
        assert config.cooldown_seconds == 60.0
        assert config.apply_modulation_to_updates is True

    def test_custom_values(self):
        config = PlasticityConfig(
            max_update_per_outcome=5,
            modulation_strength=0.8,
        )

        assert config.max_update_per_outcome == 5
        assert config.modulation_strength == 0.8


class TestPlasticityCoordinator:
    """Tests for plasticity coordinator."""

    @pytest.fixture
    def coordinator(self):
        return PlasticityCoordinator()

    @pytest.fixture
    def modulated_adapter(self):
        base = MockEmbeddingAdapter(dimension=128)
        return ModulatedEmbeddingAdapter(base)

    def test_creation(self, coordinator):
        assert coordinator is not None
        assert coordinator._update_count == 0

    def test_set_salience_provider(self, coordinator):
        provider = LearnedSalienceProvider()
        coordinator.set_salience_provider(provider)

        assert coordinator._salience_provider is provider

    def test_set_modulated_adapter(self, coordinator, modulated_adapter):
        coordinator.set_modulated_adapter(modulated_adapter)

        assert coordinator._adapter is modulated_adapter

    @pytest.mark.asyncio
    async def test_process_outcome_empty(self, coordinator):
        updates = await coordinator.process_outcome(
            outcome_score=0.8,
            retrieved_embeddings=[],
            query_embedding=np.random.randn(128).astype(np.float32),
            memory_ids=[],
        )

        assert updates == []

    @pytest.mark.asyncio
    async def test_process_outcome_with_embeddings(self, coordinator):
        query = np.random.randn(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        retrieved = [
            np.random.randn(128).astype(np.float32),
            np.random.randn(128).astype(np.float32),
        ]
        for i, emb in enumerate(retrieved):
            retrieved[i] = emb / np.linalg.norm(emb)

        updates = await coordinator.process_outcome(
            outcome_score=0.9,
            retrieved_embeddings=retrieved,
            query_embedding=query,
            memory_ids=["mem1", "mem2"],
        )

        assert len(updates) <= 2
        assert coordinator._update_count >= 0

    @pytest.mark.asyncio
    async def test_process_outcome_with_modulation(self, coordinator, modulated_adapter):
        coordinator.set_modulated_adapter(modulated_adapter)

        query = np.random.randn(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        retrieved = [np.random.randn(128).astype(np.float32)]
        retrieved[0] = retrieved[0] / np.linalg.norm(retrieved[0])

        updates = await coordinator.process_outcome(
            outcome_score=0.7,
            retrieved_embeddings=retrieved,
            query_embedding=query,
            memory_ids=["mem1"],
            current_state=NeuromodulatorState.for_retrieval(),
        )

        # Updates should be normalized
        for update in updates:
            norm = np.linalg.norm(update.updated_embedding)
            assert abs(norm - 1.0) < 1e-5

    def test_get_stats(self, coordinator):
        stats = coordinator.get_stats()

        assert "update_count" in stats
        assert "homeostatic_state" in stats
        assert "has_salience_provider" in stats
        assert "has_modulated_adapter" in stats


class TestCreatePlasticityCoordinator:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        coordinator = create_plasticity_coordinator()

        assert coordinator is not None

    def test_create_with_adapter(self):
        base = MockEmbeddingAdapter(dimension=64)
        adapter = ModulatedEmbeddingAdapter(base)

        coordinator = create_plasticity_coordinator(modulated_adapter=adapter)

        assert coordinator._adapter is adapter

    def test_create_with_config(self):
        config = PlasticityConfig(max_update_per_outcome=3)

        coordinator = create_plasticity_coordinator(config=config)

        assert coordinator._config.max_update_per_outcome == 3
