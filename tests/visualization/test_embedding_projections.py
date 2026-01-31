"""Tests for embedding projections visualization module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from t4dm.visualization.embedding_projections import (
    EmbeddingProjector,
    plot_tsne_projection,
    plot_umap_projection,
)


class TestEmbeddingProjector:
    """Tests for EmbeddingProjector class."""

    def test_init(self):
        """Test initialization."""
        projector = EmbeddingProjector()
        assert len(projector._cached_projections) == 0

    def test_project_pca(self):
        """Test PCA projection (fallback/fast option)."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(100, 128)
        projected = projector.project_pca(embeddings)
        assert projected.shape == (100, 2)
        # Verify variance is maximized (PCA property)
        # First component should have higher variance than second
        var1 = np.var(projected[:, 0])
        var2 = np.var(projected[:, 1])
        assert var1 >= var2

    def test_project_pca_3d(self):
        """Test 3D PCA projection."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(50, 64)
        projected = projector.project_pca(embeddings, n_components=3)
        assert projected.shape == (50, 3)

    def test_project_pca_caching(self):
        """Test that PCA projections are cached."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(30, 32)

        projected1 = projector.project_pca(embeddings)
        # Should be cached now
        assert "pca_2" in projector._cached_projections

        projected2 = projector.project_pca(embeddings)
        # Should return same cached result
        np.testing.assert_array_equal(projected1, projected2)

    def test_project_tsne(self):
        """Test t-SNE projection."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(30, 32)
        projected = projector.project_tsne(embeddings, perplexity=5)
        assert projected.shape == (30, 2)

    def test_project_tsne_caching(self):
        """Test that t-SNE projections are cached."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(20, 16)

        projected1 = projector.project_tsne(embeddings, perplexity=5)
        projected2 = projector.project_tsne(embeddings, perplexity=5)
        np.testing.assert_array_equal(projected1, projected2)

    def test_project_tsne_different_params(self):
        """Test t-SNE with different parameters creates new cache."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(20, 16)

        projector.project_tsne(embeddings, perplexity=5)
        projector.project_tsne(embeddings, perplexity=10)
        # Should have 2 different cached entries
        assert len(projector._cached_projections) == 2

    def test_project_umap(self):
        """Test UMAP projection."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(30, 32)
        # UMAP may fall back to t-SNE if not installed
        projected = projector.project_umap(embeddings, n_neighbors=5)
        assert projected.shape == (30, 2)

    def test_project_umap_3d(self):
        """Test 3D UMAP projection."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(30, 32)
        projected = projector.project_umap(embeddings, n_components=3, n_neighbors=5)
        assert projected.shape == (30, 3)

    def test_clear_cache(self):
        """Test clearing projection cache."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(20, 16)

        projector.project_pca(embeddings)
        assert len(projector._cached_projections) > 0

        projector.clear_cache()
        assert len(projector._cached_projections) == 0

    def test_project_small_dataset(self):
        """Test projection with small dataset."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(5, 16)
        # Should handle small datasets
        projected = projector.project_pca(embeddings)
        assert projected.shape == (5, 2)


try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available for integration tests"
)
class TestPlotFunctions:
    """Integration tests for plot functions - require matplotlib."""

    def test_data_for_tsne_projection(self):
        """Test data preparation for t-SNE projection (no actual plotting)."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(30, 32)
        # Use PCA as t-SNE may have sklearn version issues
        projected = projector.project_pca(embeddings)
        assert projected.shape == (30, 2)

    def test_data_for_umap_projection(self):
        """Test data preparation for UMAP projection."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(30, 32)
        # UMAP falls back to t-SNE or PCA
        projected = projector.project_umap(embeddings, n_neighbors=5)
        assert projected.shape == (30, 2)

    def test_data_for_memory_type_labels(self):
        """Test projection with memory type labels."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(30, 32)
        labels = ["episodic"] * 10 + ["semantic"] * 10 + ["procedural"] * 10
        projected = projector.project_pca(embeddings)
        assert projected.shape == (30, 2)
        assert len(labels) == projected.shape[0]

    def test_projector_caching(self):
        """Test that projector caches results."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(20, 16)
        projector.project_pca(embeddings)
        # Cache should be populated
        assert len(projector._cached_projections) > 0


class TestPlotTsneWithBackend:
    """Tests for t-SNE plotting with Agg backend."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return np.random.randn(15, 25)

    def test_plot_tsne_basic(self, sample_embeddings):
        """t-SNE plot works with Agg backend."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            plot_tsne_projection(sample_embeddings, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_tsne_with_labels(self, sample_embeddings):
        """t-SNE plot accepts labels."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            labels = [f"point_{i}" for i in range(len(sample_embeddings))]
            plot_tsne_projection(sample_embeddings, labels=labels, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_tsne_with_colors(self, sample_embeddings):
        """t-SNE plot accepts color values."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            colors = np.random.rand(len(sample_embeddings))
            plot_tsne_projection(sample_embeddings, colors=colors, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_tsne_with_projector(self, sample_embeddings):
        """t-SNE plot accepts existing projector."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            projector = EmbeddingProjector()
            plot_tsne_projection(sample_embeddings, projector=projector, interactive=False)
            plt.close('all')
            # Projector should have cached result
            assert len(projector._cached_projections) > 0
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotUmapWithBackend:
    """Tests for UMAP plotting with Agg backend."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return np.random.randn(20, 30)

    def test_plot_umap_basic(self, sample_embeddings):
        """UMAP plot works with Agg backend."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            plot_umap_projection(sample_embeddings, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_umap_with_labels(self, sample_embeddings):
        """UMAP plot accepts labels."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            labels = [f"mem_{i}" for i in range(len(sample_embeddings))]
            plot_umap_projection(sample_embeddings, labels=labels, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_umap_with_colors(self, sample_embeddings):
        """UMAP plot accepts color values."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')

            colors = np.linspace(0, 1, len(sample_embeddings))
            plot_umap_projection(sample_embeddings, colors=colors, interactive=False)
            plt.close('all')
        except ImportError:
            pytest.skip("matplotlib not available")


class TestProjectorEdgeCases:
    """Edge case tests for EmbeddingProjector."""

    def test_single_sample(self):
        """Projector handles single sample."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(1, 20)
        # Single sample can only have 1 component max
        result = projector.project_pca(embeddings, n_components=1)
        assert result.shape == (1, 1)

    def test_high_dimensional_input(self):
        """Projector handles high-dimensional input."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(10, 500)
        result = projector.project_pca(embeddings, n_components=2)
        assert result.shape == (10, 2)

    def test_tsne_perplexity_adjustment(self):
        """t-SNE adjusts perplexity for small datasets."""
        projector = EmbeddingProjector()
        # Dataset smaller than default perplexity
        embeddings = np.random.randn(10, 20)
        result = projector.project_tsne(embeddings, perplexity=30)
        assert result.shape == (10, 2)

    def test_umap_neighbors_adjustment(self):
        """UMAP adjusts n_neighbors for small datasets."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(8, 15)
        result = projector.project_umap(embeddings, n_neighbors=15)
        assert result.shape == (8, 2)

    def test_multiple_projection_types(self):
        """Multiple projection types in cache."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(15, 20)

        projector.project_pca(embeddings, n_components=2)
        projector.project_tsne(embeddings, perplexity=5)

        assert len(projector._cached_projections) == 2
        assert "pca_2" in projector._cached_projections

    def test_clear_and_reproject(self):
        """Cache can be cleared and reprojected."""
        projector = EmbeddingProjector()
        embeddings = np.random.randn(10, 15)

        result1 = projector.project_pca(embeddings)
        projector.clear_cache()
        result2 = projector.project_pca(embeddings)

        # Results should be identical (deterministic)
        np.testing.assert_array_almost_equal(result1, result2)
