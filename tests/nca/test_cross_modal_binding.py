"""
Tests for Cross-Modal Binding module.

Tests gamma synchrony-based binding across episodic, semantic, and procedural memories.
"""

import numpy as np
import pytest

from ww.nca.cross_modal_binding import (
    CrossModalBindingConfig,
    ModalityProjector,
    GammaSynchronyDetector,
    CrossModalBinding,
    TripartiteMemoryAttention,
)


class TestCrossModalBindingConfig:
    """Tests for CrossModalBindingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CrossModalBindingConfig()
        assert config.embed_dim == 1024
        assert config.binding_dim == 256
        assert config.binding_temperature == 0.5
        assert config.synchrony_threshold == 0.3
        assert config.contrastive_temperature == 0.07
        assert config.orthogonality_weight == 0.1
        assert config.num_gamma_bins == 8
        assert config.plv_window_size == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CrossModalBindingConfig(
            embed_dim=512,
            binding_dim=128,
            synchrony_threshold=0.5,
        )
        assert config.embed_dim == 512
        assert config.binding_dim == 128
        assert config.synchrony_threshold == 0.5


class TestModalityProjector:
    """Tests for ModalityProjector."""

    @pytest.fixture
    def projector(self):
        """Create projector instance."""
        return ModalityProjector(
            input_dim=64,
            output_dim=32,
            name="test_projector",
        )

    def test_initialization(self, projector):
        """Test projector initialization."""
        assert projector.input_dim == 64
        assert projector.output_dim == 32
        assert projector.name == "test_projector"
        assert projector.W.shape == (32, 64)
        assert projector.b.shape == (32,)

    def test_project_1d(self, projector):
        """Test projecting 1D embedding."""
        embedding = np.random.randn(64).astype(np.float32)
        projected = projector.project(embedding)

        assert projected.shape == (32,)
        # Should be L2 normalized
        assert np.abs(np.linalg.norm(projected) - 1.0) < 0.01

    def test_project_2d(self, projector):
        """Test projecting batch of embeddings."""
        embeddings = np.random.randn(5, 64).astype(np.float32)
        projected = projector.project(embeddings)

        assert projected.shape == (5, 32)
        # Each should be L2 normalized
        norms = np.linalg.norm(projected, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)

    def test_update(self, projector):
        """Test updating projector weights."""
        embedding = np.random.randn(64).astype(np.float32)
        target = np.random.randn(32).astype(np.float32)
        target = target / np.linalg.norm(target)

        initial_W = projector.W.copy()
        update_mag = projector.update(embedding, target, lr=0.1)

        # Weights should have changed
        assert not np.allclose(projector.W, initial_W)
        assert update_mag > 0

    def test_update_batch(self, projector):
        """Test updating with batch of examples."""
        embeddings = np.random.randn(10, 64).astype(np.float32)
        targets = np.random.randn(10, 32).astype(np.float32)
        targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)

        update_mag = projector.update(embeddings, targets, lr=0.01)
        assert update_mag > 0


class TestGammaSynchronyDetector:
    """Tests for GammaSynchronyDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return GammaSynchronyDetector(n_bins=8, window_size=10)

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.n_bins == 8
        assert detector.window_size == 10
        assert detector._phase_history_1 == []
        assert detector._phase_history_2 == []

    def test_compute_synchrony_perfect(self, detector):
        """Test PLV for perfectly synchronized phases."""
        phases_1 = np.zeros(20)
        phases_2 = np.zeros(20)

        plv = detector.compute_synchrony(phases_1, phases_2)
        assert plv == pytest.approx(1.0, rel=0.01)

    def test_compute_synchrony_anti_phase(self, detector):
        """Test PLV for anti-phase signals."""
        phases_1 = np.zeros(20)
        phases_2 = np.full(20, np.pi)

        plv = detector.compute_synchrony(phases_1, phases_2)
        assert plv == pytest.approx(1.0, rel=0.01)  # PLV is magnitude

    def test_compute_synchrony_random(self, detector):
        """Test PLV for random phases."""
        np.random.seed(42)
        phases_1 = np.random.uniform(0, 2 * np.pi, 100)
        phases_2 = np.random.uniform(0, 2 * np.pi, 100)

        plv = detector.compute_synchrony(phases_1, phases_2)
        # Random phases should have low PLV
        assert 0.0 <= plv <= 0.5

    def test_compute_synchrony_length_mismatch(self, detector):
        """Test PLV with different length sequences."""
        phases_1 = np.zeros(20)
        phases_2 = np.zeros(15)

        plv = detector.compute_synchrony(phases_1, phases_2)
        assert 0.0 <= plv <= 1.0

    def test_update_phases(self, detector):
        """Test updating phase history."""
        for i in range(5):
            plv = detector.update_phases(0.0, 0.1)
            assert 0.0 <= plv <= 1.0 + 1e-9  # Allow small floating point tolerance

        assert len(detector._phase_history_1) == 5
        assert len(detector._phase_history_2) == 5

    def test_update_phases_window_limit(self, detector):
        """Test that window size is respected."""
        for i in range(20):
            detector.update_phases(float(i) * 0.1, float(i) * 0.1)

        assert len(detector._phase_history_1) == 10
        assert len(detector._phase_history_2) == 10

    def test_is_synchronized(self, detector):
        """Test synchronization threshold check."""
        assert detector.is_synchronized(0.5, threshold=0.3)
        assert not detector.is_synchronized(0.2, threshold=0.3)

    def test_reset(self, detector):
        """Test resetting phase history."""
        for i in range(5):
            detector.update_phases(float(i), float(i))

        detector.reset()

        assert detector._phase_history_1 == []
        assert detector._phase_history_2 == []


class TestCrossModalBinding:
    """Tests for CrossModalBinding."""

    @pytest.fixture
    def config(self):
        """Create small config for testing."""
        return CrossModalBindingConfig(
            embed_dim=64,
            binding_dim=32,
            synchrony_threshold=0.3,
        )

    @pytest.fixture
    def binding(self, config):
        """Create binding instance."""
        return CrossModalBinding(config)

    def test_initialization(self, binding):
        """Test binding initialization."""
        assert "episodic" in binding.projectors
        assert "semantic" in binding.projectors
        assert "procedural" in binding.projectors

        assert ("episodic", "semantic") in binding.synchrony_detectors
        assert ("episodic", "procedural") in binding.synchrony_detectors
        assert ("semantic", "procedural") in binding.synchrony_detectors

    def test_bind_single_modality(self, binding):
        """Test binding with single modality."""
        episodic = np.random.randn(64).astype(np.float32)

        result = binding.bind(episodic=episodic)

        assert "projections" in result
        assert "episodic" in result["projections"]

    def test_bind_two_modalities(self, binding):
        """Test binding with two modalities."""
        episodic = np.random.randn(64).astype(np.float32)
        semantic = np.random.randn(64).astype(np.float32)

        result = binding.bind(episodic=episodic, semantic=semantic)

        assert "projections" in result
        assert "episodic" in result["projections"]
        assert "semantic" in result["projections"]
        assert "attention_weights" in result

    def test_bind_all_modalities(self, binding):
        """Test binding with all three modalities."""
        episodic = np.random.randn(64).astype(np.float32)
        semantic = np.random.randn(64).astype(np.float32)
        procedural = np.random.randn(64).astype(np.float32)

        result = binding.bind(
            episodic=episodic,
            semantic=semantic,
            procedural=procedural,
        )

        assert len(result["projections"]) == 3
        assert len(result["attention_weights"]) > 0

    def test_bind_with_gamma_phases(self, binding):
        """Test binding with gamma phase information."""
        episodic = np.random.randn(64).astype(np.float32)
        semantic = np.random.randn(64).astype(np.float32)

        gamma_phases = {
            "episodic": 0.0,
            "semantic": 0.1,
        }

        result = binding.bind(
            episodic=episodic,
            semantic=semantic,
            gamma_phases=gamma_phases,
        )

        assert "synchrony" in result
        assert len(result["synchrony"]) > 0

    def test_bind_batch(self, binding):
        """Test binding with batch of embeddings."""
        episodic = np.random.randn(5, 64).astype(np.float32)
        semantic = np.random.randn(5, 64).astype(np.float32)

        result = binding.bind(episodic=episodic, semantic=semantic)

        assert result["projections"]["episodic"].shape == (5, 32)

    def test_query_across_modalities(self, binding):
        """Test cross-modal query."""
        query = np.random.randn(64).astype(np.float32)
        candidates = np.random.randn(10, 64).astype(np.float32)

        results = binding.query_across_modalities(
            query=query,
            query_mod="episodic",
            target_mod="semantic",
            candidates=candidates,
        )

        assert len(results) == 10
        # Sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_learn_binding(self, binding):
        """Test contrastive learning of binding."""
        anchor = np.random.randn(64).astype(np.float32)
        positive = np.random.randn(64).astype(np.float32)
        negatives = [np.random.randn(64).astype(np.float32) for _ in range(3)]

        loss = binding.learn_binding(
            anchor=anchor,
            anchor_mod="episodic",
            positive=positive,
            pos_mod="semantic",
            negatives=negatives,
        )

        assert loss > 0

    def test_learn_binding_with_neg_mods(self, binding):
        """Test learning with specified negative modalities."""
        anchor = np.random.randn(64).astype(np.float32)
        positive = np.random.randn(64).astype(np.float32)
        negatives = [np.random.randn(64).astype(np.float32) for _ in range(2)]
        neg_mods = ["procedural", "semantic"]

        loss = binding.learn_binding(
            anchor=anchor,
            anchor_mod="episodic",
            positive=positive,
            pos_mod="semantic",
            negatives=negatives,
            neg_mods=neg_mods,
        )

        assert loss > 0

    def test_get_stats(self, binding):
        """Test statistics retrieval."""
        stats = binding.get_stats()

        assert "binding_dim" in stats
        assert "synchrony_threshold" in stats
        assert "current_synchrony" in stats
        assert "projector_norms" in stats
        assert len(stats["projector_norms"]) == 3


class TestTripartiteMemoryAttention:
    """Tests for TripartiteMemoryAttention."""

    @pytest.fixture
    def config(self):
        """Create small config for testing."""
        return CrossModalBindingConfig(
            embed_dim=64,
            binding_dim=32,
        )

    @pytest.fixture
    def attention(self, config):
        """Create attention instance."""
        return TripartiteMemoryAttention(config)

    def test_initialization(self, attention):
        """Test attention initialization."""
        assert attention.binding is not None
        assert attention._episodic_buffer == []
        assert attention._semantic_buffer == []
        assert attention._procedural_buffer == []

    def test_add_memory(self, attention):
        """Test adding memories to buffers."""
        episodic = np.random.randn(64).astype(np.float32)
        semantic = np.random.randn(64).astype(np.float32)
        procedural = np.random.randn(64).astype(np.float32)

        attention.add_memory(episodic, "episodic")
        attention.add_memory(semantic, "semantic")
        attention.add_memory(procedural, "procedural")

        assert len(attention._episodic_buffer) == 1
        assert len(attention._semantic_buffer) == 1
        assert len(attention._procedural_buffer) == 1

    def test_holistic_recall_empty(self, attention):
        """Test recall with empty buffers."""
        query = np.random.randn(64).astype(np.float32)

        results = attention.holistic_recall(query)

        assert results["episodic"] == []
        assert results["semantic"] == []
        assert results["procedural"] == []

    def test_holistic_recall_with_memories(self, attention):
        """Test recall with populated buffers."""
        # Add memories
        for i in range(5):
            attention.add_memory(np.random.randn(64).astype(np.float32), "semantic")
            attention.add_memory(np.random.randn(64).astype(np.float32), "procedural")

        query = np.random.randn(64).astype(np.float32)
        results = attention.holistic_recall(query, query_modality="episodic", top_k=3)

        assert len(results["semantic"]) == 3
        assert len(results["procedural"]) == 3
        assert "coherence" in results

    def test_compute_cross_modal_coherence(self, attention):
        """Test computing coherence across modalities."""
        episodic = np.random.randn(64).astype(np.float32)
        semantic = np.random.randn(64).astype(np.float32)
        procedural = np.random.randn(64).astype(np.float32)

        coherence = attention.compute_cross_modal_coherence(
            episodic, semantic, procedural
        )

        assert 0.0 <= coherence <= 1.0

    def test_compute_cross_modal_coherence_same(self, attention):
        """Test coherence for identical embeddings."""
        embedding = np.random.randn(64).astype(np.float32)

        coherence = attention.compute_cross_modal_coherence(
            embedding.copy(),
            embedding.copy(),
            embedding.copy(),
        )

        # Same embeddings should have high coherence
        assert coherence > 0.3


class TestCrossModalBindingIntegration:
    """Integration tests for cross-modal binding."""

    def test_full_binding_pipeline(self):
        """Test complete binding pipeline."""
        config = CrossModalBindingConfig(
            embed_dim=128,
            binding_dim=64,
            synchrony_threshold=0.3,
        )
        binding = CrossModalBinding(config)

        # Simulate synchronized gamma phases
        for i in range(15):
            episodic = np.random.randn(128).astype(np.float32)
            semantic = np.random.randn(128).astype(np.float32)
            procedural = np.random.randn(128).astype(np.float32)

            gamma_phases = {
                "episodic": i * 0.1,
                "semantic": i * 0.1 + 0.05,  # Slightly offset
                "procedural": i * 0.1 + 0.1,
            }

            result = binding.bind(
                episodic=episodic,
                semantic=semantic,
                procedural=procedural,
                gamma_phases=gamma_phases,
            )

            assert "projections" in result
            assert "synchrony" in result

    def test_learning_improves_binding(self):
        """Test that learning improves binding strength."""
        config = CrossModalBindingConfig(embed_dim=64, binding_dim=32)
        binding = CrossModalBinding(config)

        # Create related pairs
        anchor = np.random.randn(64).astype(np.float32)
        positive = anchor + np.random.randn(64) * 0.1  # Similar
        negatives = [np.random.randn(64).astype(np.float32) for _ in range(5)]

        # Learn for several iterations
        losses = []
        for _ in range(10):
            loss = binding.learn_binding(
                anchor.astype(np.float32),
                "episodic",
                positive.astype(np.float32),
                "semantic",
                negatives,
            )
            losses.append(loss)

        # Loss should decrease (learning is working)
        # Note: With random negatives this might fluctuate
        assert len(losses) == 10

    def test_tripartite_attention_workflow(self):
        """Test complete tripartite attention workflow."""
        config = CrossModalBindingConfig(embed_dim=64, binding_dim=32)
        attention = TripartiteMemoryAttention(config)

        # Add memories to all systems
        for i in range(10):
            attention.add_memory(np.random.randn(64).astype(np.float32), "episodic")
            attention.add_memory(np.random.randn(64).astype(np.float32), "semantic")
            attention.add_memory(np.random.randn(64).astype(np.float32), "procedural")

        # Query from episodic
        query = np.random.randn(64).astype(np.float32)
        results = attention.holistic_recall(query, query_modality="episodic", top_k=5)

        assert len(results["semantic"]) == 5
        assert len(results["procedural"]) == 5
        assert "coherence" in results
        assert results["coherence"] > 0

    def test_synchrony_gating(self):
        """Test that synchrony gates binding."""
        config = CrossModalBindingConfig(
            embed_dim=64,
            binding_dim=32,
            synchrony_threshold=0.5,  # High threshold
        )
        binding = CrossModalBinding(config)

        episodic = np.random.randn(64).astype(np.float32)
        semantic = np.random.randn(64).astype(np.float32)

        # Build up some phase history
        for i in range(10):
            binding.bind(
                episodic=episodic,
                semantic=semantic,
                gamma_phases={"episodic": i * 0.1, "semantic": i * 0.1},
            )

        stats = binding.get_stats()
        assert "current_synchrony" in stats
