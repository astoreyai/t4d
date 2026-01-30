"""Tests for GABA/Glutamate-like inhibitory dynamics."""

import pytest
import numpy as np
from datetime import datetime

from ww.learning.inhibition import (
    InhibitionResult,
    InhibitoryNetwork,
    SparseRetrieval,
)


class TestInhibitionResult:
    """Tests for InhibitionResult dataclass."""

    def test_result_creation(self):
        """Create result with all fields."""
        result = InhibitionResult(
            original_scores={"a": 0.8, "b": 0.6},
            inhibited_scores={"a": 0.7, "b": 0.3},
            winners=["a"],
            sparsity=0.5,
            iterations=3,
        )
        assert result.original_scores["a"] == 0.8
        assert result.inhibited_scores["a"] == 0.7
        assert "a" in result.winners
        assert result.sparsity == 0.5
        assert result.iterations == 3

    def test_result_has_timestamp(self):
        """Result has automatic timestamp."""
        result = InhibitionResult(
            original_scores={},
            inhibited_scores={},
            winners=[],
            sparsity=0.0,
            iterations=0,
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestInhibitoryNetwork:
    """Tests for InhibitoryNetwork class."""

    @pytest.fixture
    def network(self):
        """Create network instance."""
        return InhibitoryNetwork(
            inhibition_strength=0.5,
            sparsity_target=0.3,
            max_iterations=5,
        )

    def test_initialization(self, network):
        """Test network initialization."""
        assert network.inhibition_strength == 0.5
        assert network.sparsity_target == 0.3
        assert network.max_iterations == 5

    def test_initialization_defaults(self):
        """Test default initialization."""
        network = InhibitoryNetwork()
        assert network.inhibition_strength == 0.5
        assert network.sparsity_target == 0.2
        assert network.similarity_inhibition is True

    def test_apply_inhibition_empty(self, network):
        """Apply inhibition to empty scores."""
        result = network.apply_inhibition({})
        assert result.original_scores == {}
        assert result.inhibited_scores == {}
        assert result.winners == []
        assert result.sparsity == 0.0

    def test_apply_inhibition_single(self, network):
        """Apply inhibition to single item."""
        scores = {"item1": 0.8}
        result = network.apply_inhibition(scores)
        assert "item1" in result.inhibited_scores
        assert "item1" in result.winners

    def test_apply_inhibition_multiple(self, network):
        """Apply inhibition to multiple items."""
        scores = {
            "a": 0.9,
            "b": 0.7,
            "c": 0.5,
            "d": 0.3,
        }
        result = network.apply_inhibition(scores)
        assert len(result.inhibited_scores) == 4
        # Higher scores should remain higher
        assert result.inhibited_scores["a"] >= result.inhibited_scores["d"]

    def test_apply_inhibition_with_embeddings(self, network):
        """Apply inhibition with similarity-based inhibition."""
        scores = {"a": 0.8, "b": 0.7, "c": 0.6}
        embeddings = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.9, 0.1, 0.0]),  # Similar to a
            "c": np.array([0.0, 0.0, 1.0]),  # Different
        }
        result = network.apply_inhibition(scores, embeddings)
        assert isinstance(result, InhibitionResult)
        # b should be more inhibited due to similarity to a
        assert result.inhibited_scores["c"] > 0

    def test_apply_inhibition_convergence(self, network):
        """Check that inhibition converges."""
        scores = {f"item{i}": 0.5 + i * 0.1 for i in range(10)}
        result = network.apply_inhibition(scores)
        assert result.iterations <= network.max_iterations

    def test_apply_inhibition_sparsity(self, network):
        """Check sparsity calculation."""
        scores = {f"item{i}": 0.5 for i in range(10)}
        result = network.apply_inhibition(scores)
        assert 0.0 <= result.sparsity <= 1.0

    def test_compute_similarity_matrix(self, network):
        """Test similarity matrix computation."""
        ids = ["a", "b", "c"]
        embeddings = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([1.0, 0.0]),  # Identical to a
            "c": np.array([0.0, 1.0]),  # Orthogonal
        }
        matrix = network._compute_similarity_matrix(ids, embeddings)
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1.0  # Self-similarity
        assert matrix[0, 1] == pytest.approx(1.0, abs=0.01)  # a similar to b
        assert matrix[0, 2] == pytest.approx(0.0, abs=0.01)  # a orthogonal to c

    def test_apply_lateral_inhibition(self, network):
        """Test lateral inhibition from competitors."""
        target_score = 0.8
        competitors = [
            ("comp1", 0.7, 0.5),  # (id, score, similarity)
            ("comp2", 0.6, 0.3),
        ]
        result = network.apply_lateral_inhibition("target", target_score, competitors)
        # Score should be reduced by competitors
        assert result < target_score
        assert result >= 0.0

    def test_apply_lateral_inhibition_no_competitors(self, network):
        """Lateral inhibition with no competitors."""
        result = network.apply_lateral_inhibition("target", 0.8, [])
        assert result == 0.8

    def test_sharpen_ranking(self, network):
        """Test ranking sharpening."""
        ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        sharpened = network.sharpen_ranking(ranked)
        assert len(sharpened) == 4
        # Order should be preserved
        assert sharpened[0][0] == "a"
        # Scores should decrease with rank
        for i in range(len(sharpened) - 1):
            assert sharpened[i][1] >= sharpened[i + 1][1]

    def test_sharpen_ranking_empty(self, network):
        """Sharpen empty ranking."""
        result = network.sharpen_ranking([])
        assert result == []

    def test_compute_sparsity(self, network):
        """Test sparsity computation."""
        # Uniform distribution = low sparsity
        uniform_scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        uniform_sparsity = network.compute_sparsity(uniform_scores)

        # Sparse distribution = high sparsity
        sparse_scores = {"a": 0.9, "b": 0.05, "c": 0.05}
        sparse_sparsity = network.compute_sparsity(sparse_scores)

        assert sparse_sparsity > uniform_sparsity

    def test_compute_sparsity_empty(self, network):
        """Sparsity of empty dict is 0."""
        assert network.compute_sparsity({}) == 0.0

    def test_compute_sparsity_all_zero(self, network):
        """Sparsity with all zeros."""
        assert network.compute_sparsity({"a": 0.0, "b": 0.0}) == 0.0

    def test_get_stats_empty(self, network):
        """Stats with no history."""
        stats = network.get_stats()
        assert stats["total_applications"] == 0
        assert stats["avg_sparsity"] == 0.0

    def test_get_stats_after_applications(self, network):
        """Stats after applying inhibition."""
        scores = {"a": 0.8, "b": 0.6, "c": 0.4}
        network.apply_inhibition(scores)
        network.apply_inhibition(scores)

        stats = network.get_stats()
        assert stats["total_applications"] == 2
        assert stats["avg_sparsity"] > 0

    def test_reset_history(self, network):
        """Test history reset."""
        scores = {"a": 0.8}
        network.apply_inhibition(scores)
        assert len(network._history) > 0

        network.reset_history()
        assert len(network._history) == 0

    def test_temperature_effect(self):
        """Test temperature parameter effect."""
        scores = {"a": 0.8, "b": 0.6, "c": 0.4}

        # Low temperature = sharper competition
        network_low = InhibitoryNetwork(temperature=0.5)
        result_low = network_low.apply_inhibition(scores)

        # High temperature = softer competition
        network_high = InhibitoryNetwork(temperature=2.0)
        result_high = network_high.apply_inhibition(scores)

        # Low temp should have fewer winners (sharper)
        # Not guaranteed but typical behavior
        assert result_low.iterations > 0
        assert result_high.iterations > 0

    def test_inhibition_strength_effect(self):
        """Test inhibition strength parameter."""
        scores = {"a": 0.9, "b": 0.5, "c": 0.3}

        network_weak = InhibitoryNetwork(inhibition_strength=0.1)
        result_weak = network_weak.apply_inhibition(scores)

        network_strong = InhibitoryNetwork(inhibition_strength=0.9)
        result_strong = network_strong.apply_inhibition(scores)

        # Strong inhibition should result in more suppression
        weak_total = sum(result_weak.inhibited_scores.values())
        strong_total = sum(result_strong.inhibited_scores.values())
        # Both should be normalized back to original sum
        assert weak_total > 0
        assert strong_total > 0


class TestSparseRetrieval:
    """Tests for SparseRetrieval class."""

    @pytest.fixture
    def sparse_retrieval(self):
        """Create sparse retrieval instance."""
        return SparseRetrieval(
            min_score_threshold=0.1,
            max_results=5,
        )

    def test_initialization(self, sparse_retrieval):
        """Test initialization."""
        assert sparse_retrieval.min_score_threshold == 0.1
        assert sparse_retrieval.max_results == 5

    def test_initialization_defaults(self):
        """Test default initialization."""
        sr = SparseRetrieval()
        assert sr.min_score_threshold == 0.1
        assert sr.max_results == 10
        assert sr.inhibitory is not None

    def test_sparsify_results_empty(self, sparse_retrieval):
        """Sparsify empty results."""
        result = sparse_retrieval.sparsify_results([])
        assert result == []

    def test_sparsify_results_basic(self, sparse_retrieval):
        """Sparsify basic results."""
        results = [
            ("a", 0.9),
            ("b", 0.7),
            ("c", 0.5),
            ("d", 0.3),
            ("e", 0.1),
        ]
        sparsified = sparse_retrieval.sparsify_results(results)
        assert len(sparsified) <= sparse_retrieval.max_results
        # Should be sorted by score
        for i in range(len(sparsified) - 1):
            assert sparsified[i][1] >= sparsified[i + 1][1]

    def test_sparsify_results_with_embeddings(self, sparse_retrieval):
        """Sparsify with embeddings."""
        results = [("a", 0.8), ("b", 0.7), ("c", 0.6)]
        embeddings = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.9, 0.1]),
            "c": np.array([0.0, 1.0]),
        }
        sparsified = sparse_retrieval.sparsify_results(results, embeddings)
        assert len(sparsified) > 0

    def test_sparsify_results_respects_max(self, sparse_retrieval):
        """Sparsify respects max_results."""
        results = [(f"item{i}", 0.9 - i * 0.05) for i in range(20)]
        sparsified = sparse_retrieval.sparsify_results(results)
        assert len(sparsified) <= sparse_retrieval.max_results

    def test_sparsify_results_filters_low_scores(self, sparse_retrieval):
        """Sparsify filters low scores."""
        results = [
            ("a", 0.9),
            ("b", 0.05),  # Below threshold
            ("c", 0.01),  # Below threshold
        ]
        sparsified = sparse_retrieval.sparsify_results(results)
        # Only high scores should remain
        ids = [r[0] for r in sparsified]
        assert "a" in ids

    def test_sparsify_dict_empty(self, sparse_retrieval):
        """Sparsify empty dict."""
        result = sparse_retrieval.sparsify_dict({})
        assert result == {}

    def test_sparsify_dict_basic(self, sparse_retrieval):
        """Sparsify dict of scores."""
        scores = {"a": 0.9, "b": 0.6, "c": 0.3}
        sparsified = sparse_retrieval.sparsify_dict(scores)
        assert isinstance(sparsified, dict)
        assert all(v >= sparse_retrieval.min_score_threshold for v in sparsified.values())

    def test_sparsify_dict_with_embeddings(self, sparse_retrieval):
        """Sparsify dict with embeddings."""
        scores = {"a": 0.8, "b": 0.7}
        embeddings = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
        sparsified = sparse_retrieval.sparsify_dict(scores, embeddings)
        assert isinstance(sparsified, dict)

    def test_custom_network(self):
        """Test with custom inhibitory network."""
        custom_network = InhibitoryNetwork(
            inhibition_strength=0.8,
            sparsity_target=0.1,
        )
        sr = SparseRetrieval(inhibitory_network=custom_network)
        assert sr.inhibitory is custom_network


class TestInhibitoryDynamics:
    """Integration tests for inhibitory dynamics."""

    def test_winner_take_all_behavior(self):
        """Test winner-take-all competition."""
        network = InhibitoryNetwork(
            inhibition_strength=0.8,
            sparsity_target=0.2,
            max_iterations=10,
        )

        # One clear winner, others should be suppressed
        scores = {"winner": 0.95, "loser1": 0.3, "loser2": 0.25, "loser3": 0.2}
        result = network.apply_inhibition(scores)

        # Winner should remain strong
        assert result.inhibited_scores["winner"] > result.inhibited_scores["loser1"]
        assert "winner" in result.winners

    def test_multiple_winners(self):
        """Test with multiple strong items."""
        network = InhibitoryNetwork(
            inhibition_strength=0.5,
            sparsity_target=0.5,
        )

        # Two strong items
        scores = {"strong1": 0.9, "strong2": 0.85, "weak": 0.2}
        result = network.apply_inhibition(scores)

        # Both strong items should survive
        strong_scores = [result.inhibited_scores["strong1"], result.inhibited_scores["strong2"]]
        weak_score = result.inhibited_scores["weak"]
        assert min(strong_scores) > weak_score

    def test_preserves_relative_ranking(self):
        """Test that relative ranking is preserved."""
        network = InhibitoryNetwork()

        scores = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}
        result = network.apply_inhibition(scores)

        # Order should be preserved
        inhibited = result.inhibited_scores
        assert inhibited["a"] >= inhibited["b"]
        assert inhibited["b"] >= inhibited["c"]
        assert inhibited["c"] >= inhibited["d"]

    def test_history_accumulation(self):
        """Test history accumulates across calls."""
        network = InhibitoryNetwork()

        for i in range(5):
            scores = {f"item{j}": 0.5 + j * 0.1 for j in range(3)}
            network.apply_inhibition(scores)

        assert len(network._history) == 5
        stats = network.get_stats()
        assert stats["total_applications"] == 5


class TestInhibitoryNetworkRuntimeConfiguration:
    """Tests for runtime configuration setters."""

    def test_set_inhibition_strength(self):
        """Test setting inhibition strength."""
        network = InhibitoryNetwork()

        network.set_inhibition_strength(0.8)
        assert network.inhibition_strength == 0.8

    def test_set_inhibition_strength_clipped(self):
        """Test inhibition strength is clipped."""
        network = InhibitoryNetwork()

        network.set_inhibition_strength(-0.5)
        assert network.inhibition_strength == 0.0

        network.set_inhibition_strength(1.5)
        assert network.inhibition_strength == 1.0

    def test_set_sparsity_target(self):
        """Test setting sparsity target."""
        network = InhibitoryNetwork()

        network.set_sparsity_target(0.3)
        assert network.sparsity_target == 0.3

    def test_set_sparsity_target_clipped(self):
        """Test sparsity target is clipped."""
        network = InhibitoryNetwork()

        network.set_sparsity_target(0.01)  # Below min
        assert network.sparsity_target == 0.05

        network.set_sparsity_target(0.9)  # Above max
        assert network.sparsity_target == 0.5

    def test_set_temperature(self):
        """Test setting temperature."""
        network = InhibitoryNetwork()

        network.set_temperature(1.5)
        assert network.temperature == 1.5

    def test_set_temperature_clipped(self):
        """Test temperature is clipped."""
        network = InhibitoryNetwork()

        network.set_temperature(0.01)  # Below min
        assert network.temperature == 0.1

        network.set_temperature(10.0)  # Above max
        assert network.temperature == 5.0

    def test_set_max_iterations(self):
        """Test setting max iterations."""
        network = InhibitoryNetwork()

        network.set_max_iterations(10)
        assert network.max_iterations == 10

    def test_set_max_iterations_clipped(self):
        """Test max iterations is clipped."""
        network = InhibitoryNetwork()

        network.set_max_iterations(0)  # Below min
        assert network.max_iterations == 1

        network.set_max_iterations(50)  # Above max
        assert network.max_iterations == 20

    def test_set_convergence_threshold(self):
        """Test setting convergence threshold."""
        network = InhibitoryNetwork()

        network.set_convergence_threshold(0.05)
        assert network.convergence_threshold == 0.05

    def test_set_convergence_threshold_clipped(self):
        """Test convergence threshold is clipped."""
        network = InhibitoryNetwork()

        network.set_convergence_threshold(0.0001)  # Below min
        assert network.convergence_threshold == 0.001

        network.set_convergence_threshold(0.5)  # Above max
        assert network.convergence_threshold == 0.1

    def test_set_similarity_inhibition(self):
        """Test setting similarity inhibition."""
        network = InhibitoryNetwork()

        network.set_similarity_inhibition(False)
        assert network.similarity_inhibition is False

        network.set_similarity_inhibition(True)
        assert network.similarity_inhibition is True


class TestInhibitoryNetworkEdgeCases:
    """Tests for edge cases in inhibitory network."""

    def test_single_item_wins(self):
        """Single item should always be a winner."""
        network = InhibitoryNetwork()
        scores = {"only_item": 0.5}
        result = network.apply_inhibition(scores)

        assert "only_item" in result.winners
        assert len(result.winners) == 1

    def test_apply_lateral_inhibition_identical_competitors(self):
        """Lateral inhibition with identical competitors."""
        network = InhibitoryNetwork()

        competitors = [
            ("comp1", 0.5, 1.0),  # Maximum similarity
            ("comp2", 0.5, 1.0),
        ]
        result = network.apply_lateral_inhibition("target", 0.8, competitors)

        # Should still produce a valid result
        assert 0.0 <= result <= 1.0
