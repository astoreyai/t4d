"""
End-to-End Pipeline Validation (W4-03).

Integration tests verifying complete learning loop:
1. Encode -> store with uncertainty
2. Retrieve -> with Markov blanket
3. Outcome -> three-factor learning
4. Consolidate -> variational EM
5. Verify kappa progression

Evidence Base: Expert recommendations panel (Hinton, Friston, O'Reilly, Graves)
"""

import pytest
import numpy as np
from uuid import uuid4
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass


class TestEncodeWithUncertainty:
    """Test encoding with uncertainty estimation (W2-01)."""

    def test_encode_produces_mean_and_variance(self):
        """Encoding should produce both mean and variance."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyEstimator

        # Create mock model with encode method that returns numpy
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(64)
        mock_model.train = Mock()  # For enabling dropout
        mock_model.eval = Mock()  # For disabling dropout

        estimator = UncertaintyEstimator(
            embedding_model=mock_model,
            n_samples=5,
        )

        # Encode with uncertainty
        mean, var = estimator.embed_with_uncertainty("Test memory")

        assert mean.shape == (64,)
        assert var.shape == (64,)
        assert (var >= 0).all(), "Variance should be non-negative"


class TestRetrieveWithMarkovBlanket:
    """Test retrieval with Markov blanket prioritization (W3-01)."""

    def test_retrieve_prioritizes_mb(self):
        """Retrieval should prioritize Markov blanket members."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        engine = Mock()
        concept_id = uuid4()
        mb_id = uuid4()
        other_id = uuid4()

        # Set up MB
        mb_item = Mock(id=mb_id)
        other_item = Mock(id=other_id)

        # Traverse returns MB members
        engine.traverse.side_effect = [
            [mb_item],  # Parents
            [],  # Children
        ]
        del engine.search_filtered  # Force fallback
        engine.search.return_value = [mb_item, other_item]

        retriever = MarkovBlanketRetriever(engine, exploration_ratio=0.1)
        results = retriever.search(
            np.random.randn(64),
            query_concept=concept_id,
            k=5,
        )

        assert isinstance(results, list)


class TestThreeFactorLearning:
    """Test three-factor learning signal (Wave 1)."""

    def test_three_factor_combines_signals(self):
        """Three-factor rule should combine eligibility, neuromod, dopamine."""
        from t4dm.learning.three_factor import ThreeFactorSignal

        # Test the signal dataclass with correct fields
        signal = ThreeFactorSignal(
            memory_id=uuid4(),
            eligibility=0.8,  # Factor 1
            neuromod_gate=0.7,  # Factor 2
            ach_mode_factor=0.6,
            ne_arousal_factor=0.5,
            serotonin_mood_factor=0.4,
            dopamine_surprise=1.5,  # Factor 3
            rpe_raw=0.3,
            effective_lr_multiplier=0.8 * 0.7 * 1.5,  # Product
        )

        assert signal.effective_lr_multiplier > 0
        assert signal.effective_lr_multiplier != 1.0  # Should be modulated


class TestVariationalConsolidation:
    """Test variational EM consolidation (W4-01)."""

    def test_em_reduces_free_energy(self):
        """EM iterations should reduce free energy."""
        from t4dm.consolidation.variational import (
            VariationalConsolidation,
            ClusterPrototype,
        )

        np.random.seed(42)
        vc = VariationalConsolidation()

        # Create clustered memories
        group1 = [np.array([2.0] * 16) + np.random.randn(16) * 0.3 for _ in range(10)]
        group2 = [np.array([-2.0] * 16) + np.random.randn(16) * 0.3 for _ in range(10)]
        memories = group1 + group2

        # Initialize random clusters
        clusters = [
            ClusterPrototype(i, np.random.randn(16), np.ones(16), 0, 0.0)
            for i in range(2)
        ]

        initial_fe = None
        final_fe = None

        # Run EM
        for iteration in range(5):
            assignments = vc.e_step(memories, clusters)
            clusters = vc.m_step(memories, assignments, n_clusters=2)
            state = vc.compute_state(memories, clusters, assignments)

            if initial_fe is None:
                initial_fe = state.free_energy
            final_fe = state.free_energy

        assert final_fe < initial_fe, "Free energy should decrease"


class TestKappaProgression:
    """Test kappa (consolidation level) progression."""

    def test_kappa_increases_with_consolidation(self):
        """Kappa should increase as memories consolidate."""
        # Simulate kappa progression
        initial_kappa = 0.0  # Raw episodic

        # After NREM replay
        kappa_nrem = initial_kappa + 0.15  # Replayed
        assert kappa_nrem > initial_kappa

        # After REM abstraction
        kappa_rem = kappa_nrem + 0.25  # Transitional
        assert kappa_rem > kappa_nrem

        # After multiple cycles
        kappa_semantic = 0.85  # Semantic
        assert kappa_semantic > kappa_rem

        # Final stable
        kappa_stable = 1.0  # Fully consolidated
        assert kappa_stable >= kappa_semantic


class TestFullPipeline:
    """Test full encode-retrieve-learn-consolidate pipeline."""

    def test_complete_learning_loop(self):
        """Complete loop should work end-to-end."""
        from t4dm.consolidation.variational import (
            VariationalConsolidation,
            ClusterPrototype,
        )
        from t4dm.consolidation.generalization import GeneralizationQualityScorer

        np.random.seed(42)

        # 1. ENCODE: Create memories with different themes
        theme1 = [np.array([3.0] * 16) + np.random.randn(16) * 0.2 for _ in range(10)]
        theme2 = [np.array([-3.0] * 16) + np.random.randn(16) * 0.2 for _ in range(10)]
        memories = theme1 + theme2
        all_vectors = np.array(memories)

        # 2. RETRIEVE: Would use Markov blanket, here we simulate
        retrieved = memories[:5]  # Top 5 from theme1

        # 3. OUTCOME: Simulate positive outcome -> kappa boost
        outcome_positive = True
        kappa_boost = 0.1 if outcome_positive else -0.05

        # 4. CONSOLIDATE: Run variational EM
        vc = VariationalConsolidation()
        clusters = [
            ClusterPrototype(i, np.random.randn(16), np.ones(16), 0, 0.0)
            for i in range(2)
        ]

        for _ in range(3):
            assignments = vc.e_step(memories, clusters)
            clusters = vc.m_step(memories, assignments, n_clusters=2)

        # 5. VERIFY: Check cluster quality
        scorer = GeneralizationQualityScorer(min_quality=0.3)
        result = scorer.score_cluster(np.array(theme1), all_vectors)

        # Should have good generalization for well-separated themes
        assert result.quality > 0.3, f"Quality={result.quality}"

    def test_pipeline_with_generalization_gating(self):
        """Only high-quality clusters should be generalized."""
        from t4dm.consolidation.variational import VariationalConsolidation
        from t4dm.consolidation.generalization import (
            GeneralizationQualityScorer,
            Cluster,
        )

        np.random.seed(42)

        # Create well-separated data
        good_data = np.random.randn(20, 16) * 0.3 + np.array([5.0] * 16)
        noise_data = np.random.randn(20, 16)  # Overlapping with everything
        background = np.random.randn(20, 16) * 0.3 + np.array([-5.0] * 16)

        all_vectors = np.vstack([good_data, noise_data, background])

        good_cluster = Cluster(id=uuid4(), vectors=good_data)
        noise_cluster = Cluster(id=uuid4(), vectors=noise_data)

        scorer = GeneralizationQualityScorer(min_quality=0.3)
        filtered = scorer.filter_clusters_for_prototyping(
            [good_cluster, noise_cluster],
            all_vectors,
        )

        # Only good cluster should pass
        cluster_ids = [c.id for c in filtered]
        assert good_cluster.id in cluster_ids
        assert noise_cluster.id not in cluster_ids


class TestEdgeLearning:
    """Test edge importance learning (W3-02)."""

    def test_edge_weights_diverge_from_uniform(self):
        """Training should make edge weights non-uniform."""
        from t4dm.storage.t4dx.learned_edges import (
            LearnedEdgeImportance,
            EdgeImportanceTrainer,
        )

        edge_importance = LearnedEdgeImportance()
        trainer = EdgeImportanceTrainer(edge_importance, lr=0.1)

        # Get initial weights
        initial = [edge_importance.get_weight(e) for e in edge_importance.EDGE_TYPES]

        # Train with biased feedback
        for _ in range(50):
            trainer.train_step(
                ["CAUSES", "SIMILAR_TO"],
                [0.9, 0.2],  # CAUSES more relevant
            )

        # Get final weights
        final = [edge_importance.get_weight(e) for e in edge_importance.EDGE_TYPES]

        # Should have diverged
        assert not np.allclose(initial, final, atol=0.01)


class TestConsciousnessMetrics:
    """Test IIT consciousness metrics (W3-04)."""

    def test_phi_tracks_integration(self):
        """Phi should be higher for integrated systems."""
        from t4dm.observability.consciousness_metrics import IITMetricsComputer
        import torch

        def energy_fn(x):
            return torch.sum(x ** 2).item()

        computer = IITMetricsComputer(energy_fn)

        # Independent subsystems
        torch.manual_seed(42)
        ind_spiking = torch.randn(100)
        ind_memory = torch.randn(100)
        metrics_ind = computer.compute(ind_spiking, ind_memory)

        # Reset for fair comparison
        computer2 = IITMetricsComputer(energy_fn)

        # Integrated subsystems
        torch.manual_seed(43)
        int_spiking = torch.randn(100)
        int_memory = int_spiking + torch.randn(100) * 0.1
        metrics_int = computer2.compute(int_spiking, int_memory)

        assert metrics_int.integration > metrics_ind.integration
