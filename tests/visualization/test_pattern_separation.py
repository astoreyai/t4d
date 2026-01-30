"""Tests for pattern separation visualization module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ww.visualization.pattern_separation import (
    PatternSeparationVisualizer,
    plot_separation_comparison,
    plot_sparsity_distribution,
)


class TestPatternSeparationVisualizer:
    """Tests for PatternSeparationVisualizer class."""

    def test_compute_similarity_matrix_cosine(self):
        """Test computing cosine similarity matrix."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        sim = PatternSeparationVisualizer.compute_similarity_matrix(
            embeddings, normalize=True
        )
        # Diagonal should be 1.0
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[1, 1] == pytest.approx(1.0)
        # Orthogonal vectors should have 0 similarity
        assert sim[0, 1] == pytest.approx(0.0)

    def test_compute_similarity_matrix_dot_product(self):
        """Test computing dot product similarity matrix."""
        embeddings = np.array([
            [2.0, 0.0],
            [0.0, 3.0],
        ])
        sim = PatternSeparationVisualizer.compute_similarity_matrix(
            embeddings, normalize=False
        )
        assert sim[0, 0] == pytest.approx(4.0)  # 2*2
        assert sim[1, 1] == pytest.approx(9.0)  # 3*3
        assert sim[0, 1] == pytest.approx(0.0)

    def test_compute_similarity_with_zero_vector(self):
        """Test handling zero vectors in similarity computation."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 0.0],  # Zero vector
        ])
        sim = PatternSeparationVisualizer.compute_similarity_matrix(
            embeddings, normalize=True
        )
        # Should handle gracefully, not produce NaN
        # Zero vector normalized stays zero, so similarity is 0
        assert not np.isnan(sim).any()

    def test_compute_sparsity_dense(self):
        """Test sparsity computation for dense vector."""
        embedding = np.array([0.5, 0.8, 0.3, 0.9, 0.7])
        sparsity = PatternSeparationVisualizer.compute_sparsity(embedding, threshold=0.01)
        assert sparsity == pytest.approx(0.0)  # No zeros

    def test_compute_sparsity_sparse(self):
        """Test sparsity computation for sparse vector."""
        embedding = np.array([0.0, 0.0, 0.5, 0.0, 0.8])
        sparsity = PatternSeparationVisualizer.compute_sparsity(embedding, threshold=0.01)
        assert sparsity == pytest.approx(0.6)  # 3 out of 5 are zero

    def test_compute_sparsity_all_zero(self):
        """Test sparsity of all-zero vector."""
        embedding = np.zeros(10)
        sparsity = PatternSeparationVisualizer.compute_sparsity(embedding)
        assert sparsity == pytest.approx(1.0)

    def test_compute_sparsity_custom_threshold(self):
        """Test sparsity with custom threshold."""
        embedding = np.array([0.001, 0.005, 0.02, 0.1])
        sparsity = PatternSeparationVisualizer.compute_sparsity(embedding, threshold=0.01)
        assert sparsity == pytest.approx(0.5)  # 2 below threshold

    def test_analyze_separation_basic(self):
        """Test basic separation analysis."""
        original = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
        ])
        separated = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        metrics = PatternSeparationVisualizer.analyze_separation(original, separated)

        assert "mean_similarity_before" in metrics
        assert "mean_similarity_after" in metrics
        assert "similarity_reduction" in metrics
        # After separation to orthogonal, similarity should be lower
        assert metrics["mean_similarity_after"] < metrics["mean_similarity_before"]

    def test_analyze_separation_identical(self):
        """Test separation when original and separated are identical."""
        embeddings = np.random.randn(5, 10)
        metrics = PatternSeparationVisualizer.analyze_separation(embeddings, embeddings)
        assert metrics["similarity_reduction"] == pytest.approx(0.0, abs=0.01)
        assert metrics["mean_separation_magnitude"] == pytest.approx(0.0, abs=0.01)

    def test_analyze_separation_metrics_keys(self):
        """Test that all expected metrics are returned."""
        original = np.random.randn(10, 20)
        separated = np.random.randn(10, 20)
        metrics = PatternSeparationVisualizer.analyze_separation(original, separated)

        expected_keys = [
            "mean_similarity_before",
            "mean_similarity_after",
            "similarity_reduction",
            "mean_separation_magnitude",
            "mean_sparsity_before",
            "mean_sparsity_after",
            "sparsity_increase",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_analyze_separation_sparsity_change(self):
        """Test that sparsity change is computed correctly."""
        # Dense original
        original = np.ones((5, 10))
        # Sparse separated
        separated = np.zeros((5, 10))
        separated[:, :2] = 1.0  # Only first 2 dims non-zero

        metrics = PatternSeparationVisualizer.analyze_separation(original, separated)
        # Sparsity should increase
        assert metrics["sparsity_increase"] > 0


class TestPlotFunctions:
    """Tests for plot functions with mocked backends."""

    def test_data_for_separation_comparison(self):
        """Test data preparation for separation comparison."""
        original = np.random.randn(10, 20)
        separated = np.random.randn(10, 20)
        orig_sim = PatternSeparationVisualizer.compute_similarity_matrix(original)
        sep_sim = PatternSeparationVisualizer.compute_similarity_matrix(separated)
        assert orig_sim.shape == (10, 10)
        assert sep_sim.shape == (10, 10)

    def test_data_for_sparsity_distribution(self):
        """Test data preparation for sparsity distribution plot."""
        original = np.random.randn(20, 50)
        separated = np.random.randn(20, 50) * 0.1
        orig_sparsity = [PatternSeparationVisualizer.compute_sparsity(e) for e in original]
        sep_sparsity = [PatternSeparationVisualizer.compute_sparsity(e) for e in separated]
        assert len(orig_sparsity) == 20
        assert len(sep_sparsity) == 20

    def test_data_for_small_dataset(self):
        """Test data from small dataset."""
        original = np.random.randn(3, 10)
        separated = np.random.randn(3, 10)
        metrics = PatternSeparationVisualizer.analyze_separation(original, separated)
        assert "mean_similarity_before" in metrics
        assert "mean_similarity_after" in metrics

    def test_plot_functions_exist(self):
        """Plot functions are importable and callable."""
        # Verify functions are properly defined
        assert callable(plot_separation_comparison)
        assert callable(plot_sparsity_distribution)

    def test_plot_separation_comparison_with_agg(self):
        """Separation comparison runs with Agg backend."""
        original = np.random.randn(5, 10)
        separated = np.random.randn(5, 10)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            # Should not raise
            plot_separation_comparison(original, separated, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_sparsity_distribution_with_agg(self):
        """Sparsity distribution runs with Agg backend."""
        original = np.random.randn(10, 20)
        separated = np.random.randn(10, 20)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            plot_sparsity_distribution(original, separated, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotSeparationComparison:
    """Tests for plot_separation_comparison function."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        original = np.random.randn(10, 20)
        # Create separated that's more sparse
        separated = original.copy()
        separated[separated < 0.5] = 0
        return original, separated

    def test_plot_separation_comparison_matplotlib(self, sample_embeddings):
        """Test separation comparison with matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original, separated = sample_embeddings
        plot_separation_comparison(original, separated, interactive=False)
        plt.close("all")

    def test_plot_separation_comparison_save(self, sample_embeddings, tmp_path):
        """Test saving separation comparison to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original, separated = sample_embeddings
        save_path = tmp_path / "separation_comparison.png"
        plot_separation_comparison(original, separated, save_path=save_path)
        assert save_path.exists()
        plt.close("all")

    def test_plot_separation_comparison_large(self):
        """Test with larger dataset."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original = np.random.randn(50, 100)
        separated = np.random.randn(50, 100)
        plot_separation_comparison(original, separated)
        plt.close("all")

    def test_plot_separation_comparison_small(self):
        """Test with very small dataset."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original = np.random.randn(2, 5)
        separated = np.random.randn(2, 5)
        plot_separation_comparison(original, separated)
        plt.close("all")


class TestPlotSparsityDistribution:
    """Tests for plot_sparsity_distribution function."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings with varying sparsity."""
        np.random.seed(42)
        # Dense original
        original = np.random.randn(20, 50)
        # Sparse separated
        separated = np.zeros((20, 50))
        for i in range(20):
            indices = np.random.choice(50, 10, replace=False)
            separated[i, indices] = np.random.randn(10)
        return original, separated

    def test_plot_sparsity_distribution_matplotlib(self, sample_embeddings):
        """Test sparsity distribution with matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original, separated = sample_embeddings
        plot_sparsity_distribution(original, separated, interactive=False)
        plt.close("all")

    def test_plot_sparsity_distribution_save(self, sample_embeddings, tmp_path):
        """Test saving sparsity distribution to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original, separated = sample_embeddings
        save_path = tmp_path / "sparsity_distribution.png"
        plot_sparsity_distribution(original, separated, save_path=save_path)
        assert save_path.exists()
        plt.close("all")

    def test_plot_sparsity_distribution_uniform(self):
        """Test with uniform sparsity distribution."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        # All embeddings have same sparsity
        original = np.ones((10, 20))
        separated = np.zeros((10, 20))
        plot_sparsity_distribution(original, separated)
        plt.close("all")


class TestPatternSeparationEdgeCases:
    """Edge case tests for pattern separation visualization."""

    def test_single_embedding(self):
        """Test with single embedding."""
        embedding = np.random.randn(1, 10)
        sim = PatternSeparationVisualizer.compute_similarity_matrix(embedding)
        assert sim.shape == (1, 1)
        assert sim[0, 0] == pytest.approx(1.0)

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings."""
        embeddings = np.random.randn(5, 1000)
        sim = PatternSeparationVisualizer.compute_similarity_matrix(embeddings)
        assert sim.shape == (5, 5)
        assert np.all(np.isfinite(sim))

    def test_identical_embeddings(self):
        """Test with all identical embeddings."""
        base = np.random.randn(1, 20)
        embeddings = np.repeat(base, 5, axis=0)
        sim = PatternSeparationVisualizer.compute_similarity_matrix(embeddings)
        # All similarities should be 1.0
        assert np.allclose(sim, 1.0)

    def test_orthogonal_embeddings(self):
        """Test with perfectly orthogonal embeddings."""
        embeddings = np.eye(4)
        sim = PatternSeparationVisualizer.compute_similarity_matrix(embeddings)
        # Off-diagonal should be 0
        for i in range(4):
            for j in range(4):
                if i == j:
                    assert sim[i, j] == pytest.approx(1.0)
                else:
                    assert sim[i, j] == pytest.approx(0.0)

    def test_negative_embeddings(self):
        """Test with negative embedding values."""
        embeddings = np.random.randn(5, 10) * -1
        sim = PatternSeparationVisualizer.compute_similarity_matrix(embeddings)
        # Negative values should still produce valid similarities
        assert np.all(np.isfinite(sim))

    def test_separation_with_perfect_orthogonalization(self):
        """Test separation that achieves perfect orthogonalization."""
        original = np.random.randn(4, 10)
        separated = np.eye(4, 10)  # Orthogonal rows

        metrics = PatternSeparationVisualizer.analyze_separation(original, separated)
        # After orthogonalization, similarity should be much lower
        assert metrics["mean_similarity_after"] < 0.1

    def test_plot_with_zero_sparsity(self):
        """Test plotting with zero sparsity (all values non-zero)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original = np.random.randn(10, 20) + 1.0  # All positive
        separated = np.random.randn(10, 20) + 1.0
        plot_sparsity_distribution(original, separated)
        plt.close("all")

    def test_plot_with_high_sparsity(self):
        """Test plotting with very high sparsity."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        original = np.zeros((10, 100))
        original[:, 0] = 1.0  # Only one non-zero per row
        separated = np.zeros((10, 100))

        plot_sparsity_distribution(original, separated)
        plt.close("all")

    def test_analyze_many_embeddings(self):
        """Test analyze with many embeddings."""
        original = np.random.randn(100, 50)
        separated = np.random.randn(100, 50) * 0.1  # More concentrated

        metrics = PatternSeparationVisualizer.analyze_separation(original, separated)
        assert isinstance(metrics, dict)
        assert len(metrics) == 7

    def test_sparsity_threshold_effect(self):
        """Test effect of sparsity threshold."""
        embedding = np.array([0.001, 0.009, 0.011, 0.1])

        # Lower threshold
        s1 = PatternSeparationVisualizer.compute_sparsity(embedding, threshold=0.005)
        # Higher threshold
        s2 = PatternSeparationVisualizer.compute_sparsity(embedding, threshold=0.01)
        # Even higher
        s3 = PatternSeparationVisualizer.compute_sparsity(embedding, threshold=0.05)

        # Higher threshold should find more "sparse" entries
        assert s1 <= s2 <= s3
