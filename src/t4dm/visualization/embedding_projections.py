"""
Embedding projection visualization for T4DM.

Projects high-dimensional memory embeddings to 2D/3D using:
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- PCA (Principal Component Analysis)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProjector:
    """
    Projects high-dimensional embeddings to 2D/3D for visualization.

    Supports multiple dimensionality reduction techniques and caching
    of projections for interactive exploration.
    """

    def __init__(self):
        """Initialize embedding projector."""
        self._cached_projections: dict[str, np.ndarray] = {}

    def project_tsne(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Project embeddings using t-SNE.

        Args:
            embeddings: High-dimensional embeddings (N x D)
            n_components: Target dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed for reproducibility

        Returns:
            Projected embeddings (N x n_components)
        """
        cache_key = f"tsne_{n_components}_{perplexity}_{random_state}"

        if cache_key in self._cached_projections:
            logger.debug(f"Using cached t-SNE projection: {cache_key}")
            return self._cached_projections[cache_key]

        try:
            from sklearn.manifold import TSNE

            logger.info(f"Computing t-SNE projection (n={len(embeddings)}, d={embeddings.shape[1]})")

            tsne = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(embeddings) - 1),
                random_state=random_state,
                max_iter=1000
            )

            projection = tsne.fit_transform(embeddings)
            self._cached_projections[cache_key] = projection

            return projection

        except ImportError:
            logger.error("scikit-learn not available for t-SNE")
            # Fallback to PCA
            return self.project_pca(embeddings, n_components)

    def project_umap(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Project embeddings using UMAP.

        Args:
            embeddings: High-dimensional embeddings (N x D)
            n_components: Target dimensions (2 or 3)
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points
            random_state: Random seed for reproducibility

        Returns:
            Projected embeddings (N x n_components)
        """
        cache_key = f"umap_{n_components}_{n_neighbors}_{min_dist}_{random_state}"

        if cache_key in self._cached_projections:
            logger.debug(f"Using cached UMAP projection: {cache_key}")
            return self._cached_projections[cache_key]

        try:
            import umap

            logger.info(f"Computing UMAP projection (n={len(embeddings)}, d={embeddings.shape[1]})")

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, len(embeddings) - 1),
                min_dist=min_dist,
                random_state=random_state
            )

            projection = reducer.fit_transform(embeddings)
            self._cached_projections[cache_key] = projection

            return projection

        except ImportError:
            logger.warning("UMAP not available, falling back to t-SNE")
            return self.project_tsne(embeddings, n_components)

    def project_pca(
        self,
        embeddings: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """
        Project embeddings using PCA.

        Args:
            embeddings: High-dimensional embeddings (N x D)
            n_components: Target dimensions

        Returns:
            Projected embeddings (N x n_components)
        """
        cache_key = f"pca_{n_components}"

        if cache_key in self._cached_projections:
            logger.debug(f"Using cached PCA projection: {cache_key}")
            return self._cached_projections[cache_key]

        try:
            from sklearn.decomposition import PCA

            logger.info(f"Computing PCA projection (n={len(embeddings)}, d={embeddings.shape[1]})")

            pca = PCA(n_components=n_components)
            projection = pca.fit_transform(embeddings)

            logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

            self._cached_projections[cache_key] = projection

            return projection

        except ImportError:
            logger.error("scikit-learn not available for PCA")
            # Last resort: return first n dimensions
            return embeddings[:, :n_components]

    def clear_cache(self) -> None:
        """Clear cached projections."""
        self._cached_projections.clear()


def plot_tsne_projection(
    embeddings: np.ndarray,
    labels: list[str] | None = None,
    colors: np.ndarray | None = None,
    save_path: Path | None = None,
    interactive: bool = False,
    projector: EmbeddingProjector | None = None
) -> None:
    """
    Plot t-SNE projection of embeddings.

    Args:
        embeddings: High-dimensional embeddings (N x D)
        labels: Optional labels for each point
        colors: Optional color values for each point
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
        projector: Optional EmbeddingProjector instance (created if None)
    """
    if projector is None:
        projector = EmbeddingProjector()

    projection = projector.project_tsne(embeddings)

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            if colors is not None:
                scatter = go.Scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Value")
                    ),
                    text=labels if labels else None,
                    hoverinfo="text"
                )
            else:
                scatter = go.Scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode="markers",
                    marker=dict(size=8, color="blue"),
                    text=labels if labels else None,
                    hoverinfo="text"
                )

            fig.add_trace(scatter)

            fig.update_layout(
                title="t-SNE Projection of Memory Embeddings",
                xaxis_title="t-SNE 1",
                yaxis_title="t-SNE 2",
                hovermode="closest",
                height=600,
                width=800
            )

            if save_path:
                fig.write_html(str(save_path))
            else:
                fig.show()

        except ImportError:
            logger.warning("Plotly not available, falling back to matplotlib")
            interactive = False

    if not interactive:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))

            if colors is not None:
                scatter = ax.scatter(
                    projection[:, 0],
                    projection[:, 1],
                    c=colors,
                    cmap="viridis",
                    s=50,
                    alpha=0.6
                )
                plt.colorbar(scatter, ax=ax, label="Value")
            else:
                ax.scatter(
                    projection[:, 0],
                    projection[:, 1],
                    c="blue",
                    s=50,
                    alpha=0.6
                )

            # Add labels if provided
            if labels:
                for i, label in enumerate(labels[:50]):  # Limit to first 50
                    ax.annotate(
                        label[:8],
                        (projection[i, 0], projection[i, 1]),
                        fontsize=8,
                        alpha=0.7
                    )

            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_title("t-SNE Projection of Memory Embeddings")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


def plot_umap_projection(
    embeddings: np.ndarray,
    labels: list[str] | None = None,
    colors: np.ndarray | None = None,
    save_path: Path | None = None,
    interactive: bool = False,
    projector: EmbeddingProjector | None = None
) -> None:
    """
    Plot UMAP projection of embeddings.

    Args:
        embeddings: High-dimensional embeddings (N x D)
        labels: Optional labels for each point
        colors: Optional color values for each point
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
        projector: Optional EmbeddingProjector instance (created if None)
    """
    if projector is None:
        projector = EmbeddingProjector()

    projection = projector.project_umap(embeddings)

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            if colors is not None:
                scatter = go.Scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Value")
                    ),
                    text=labels if labels else None,
                    hoverinfo="text"
                )
            else:
                scatter = go.Scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode="markers",
                    marker=dict(size=8, color="green"),
                    text=labels if labels else None,
                    hoverinfo="text"
                )

            fig.add_trace(scatter)

            fig.update_layout(
                title="UMAP Projection of Memory Embeddings",
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                hovermode="closest",
                height=600,
                width=800
            )

            if save_path:
                fig.write_html(str(save_path))
            else:
                fig.show()

        except ImportError:
            logger.warning("Plotly not available, falling back to matplotlib")
            interactive = False

    if not interactive:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))

            if colors is not None:
                scatter = ax.scatter(
                    projection[:, 0],
                    projection[:, 1],
                    c=colors,
                    cmap="viridis",
                    s=50,
                    alpha=0.6
                )
                plt.colorbar(scatter, ax=ax, label="Value")
            else:
                ax.scatter(
                    projection[:, 0],
                    projection[:, 1],
                    c="green",
                    s=50,
                    alpha=0.6
                )

            # Add labels if provided
            if labels:
                for i, label in enumerate(labels[:50]):  # Limit to first 50
                    ax.annotate(
                        label[:8],
                        (projection[i, 0], projection[i, 1]),
                        fontsize=8,
                        alpha=0.7
                    )

            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("UMAP Projection of Memory Embeddings")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


__all__ = [
    "EmbeddingProjector",
    "plot_tsne_projection",
    "plot_umap_projection",
]
