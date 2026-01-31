"""
Pattern separation visualization for World Weaver.

Visualizes the effects of dentate gyrus pattern separation:
- Before/after orthogonalization
- Sparsity distributions
- Separation magnitude vs similarity
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PatternSeparationVisualizer:
    """
    Visualizes pattern separation dynamics.

    Analyzes and displays the effects of DG-style pattern separation
    on embedding distributions.
    """

    @staticmethod
    def compute_similarity_matrix(
        embeddings: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            embeddings: Array of embeddings (N x D)
            normalize: If True, use cosine similarity; else dot product

        Returns:
            Similarity matrix (N x N)
        """
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = embeddings / norms
            return normalized @ normalized.T
        return embeddings @ embeddings.T

    @staticmethod
    def compute_sparsity(embedding: np.ndarray, threshold: float = 0.01) -> float:
        """
        Compute sparsity of embedding.

        Args:
            embedding: Single embedding vector
            threshold: Values below this are considered zero

        Returns:
            Sparsity ratio [0, 1] (fraction of near-zero elements)
        """
        return float(np.mean(np.abs(embedding) < threshold))

    @staticmethod
    def analyze_separation(
        original_embeddings: np.ndarray,
        separated_embeddings: np.ndarray
    ) -> dict:
        """
        Analyze separation statistics.

        Args:
            original_embeddings: Original embeddings (N x D)
            separated_embeddings: After pattern separation (N x D)

        Returns:
            Dict with separation metrics
        """
        # Compute similarity matrices
        sim_before = PatternSeparationVisualizer.compute_similarity_matrix(
            original_embeddings
        )
        sim_after = PatternSeparationVisualizer.compute_similarity_matrix(
            separated_embeddings
        )

        # Extract off-diagonal (pairwise) similarities
        N = len(original_embeddings)
        mask = ~np.eye(N, dtype=bool)

        sim_before_pairs = sim_before[mask]
        sim_after_pairs = sim_after[mask]

        # Compute separation magnitude
        sep_magnitudes = np.linalg.norm(
            separated_embeddings - original_embeddings,
            axis=1
        )

        # Compute sparsity
        sparsity_before = [
            PatternSeparationVisualizer.compute_sparsity(emb)
            for emb in original_embeddings
        ]
        sparsity_after = [
            PatternSeparationVisualizer.compute_sparsity(emb)
            for emb in separated_embeddings
        ]

        return {
            "mean_similarity_before": float(np.mean(sim_before_pairs)),
            "mean_similarity_after": float(np.mean(sim_after_pairs)),
            "similarity_reduction": float(
                np.mean(sim_before_pairs) - np.mean(sim_after_pairs)
            ),
            "mean_separation_magnitude": float(np.mean(sep_magnitudes)),
            "mean_sparsity_before": float(np.mean(sparsity_before)),
            "mean_sparsity_after": float(np.mean(sparsity_after)),
            "sparsity_increase": float(
                np.mean(sparsity_after) - np.mean(sparsity_before)
            )
        }


def plot_separation_comparison(
    original_embeddings: np.ndarray,
    separated_embeddings: np.ndarray,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot before/after comparison of pattern separation.

    Args:
        original_embeddings: Original embeddings (N x D)
        separated_embeddings: After separation (N x D)
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    vis = PatternSeparationVisualizer()
    sim_before = vis.compute_similarity_matrix(original_embeddings)
    sim_after = vis.compute_similarity_matrix(separated_embeddings)

    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Before Separation", "After Separation")
            )

            # Before heatmap
            fig.add_trace(
                go.Heatmap(
                    z=sim_before,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(x=0.45)
                ),
                row=1, col=1
            )

            # After heatmap
            fig.add_trace(
                go.Heatmap(
                    z=sim_after,
                    colorscale="Viridis",
                    showscale=True
                ),
                row=1, col=2
            )

            fig.update_xaxes(title_text="Memory Index", row=1, col=1)
            fig.update_xaxes(title_text="Memory Index", row=1, col=2)
            fig.update_yaxes(title_text="Memory Index", row=1, col=1)
            fig.update_yaxes(title_text="Memory Index", row=1, col=2)

            fig.update_layout(
                title="Pattern Separation: Similarity Matrix Comparison",
                height=500
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Before heatmap
            im1 = ax1.imshow(sim_before, cmap="viridis", aspect="auto")
            ax1.set_xlabel("Memory Index")
            ax1.set_ylabel("Memory Index")
            ax1.set_title("Before Separation")
            plt.colorbar(im1, ax=ax1)

            # After heatmap
            im2 = ax2.imshow(sim_after, cmap="viridis", aspect="auto")
            ax2.set_xlabel("Memory Index")
            ax2.set_ylabel("Memory Index")
            ax2.set_title("After Separation")
            plt.colorbar(im2, ax=ax2)

            fig.suptitle("Pattern Separation: Similarity Matrix Comparison")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


def plot_sparsity_distribution(
    original_embeddings: np.ndarray,
    separated_embeddings: np.ndarray,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot distribution of sparsity before and after separation.

    Args:
        original_embeddings: Original embeddings (N x D)
        separated_embeddings: After separation (N x D)
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    vis = PatternSeparationVisualizer()

    sparsity_before = [vis.compute_sparsity(emb) for emb in original_embeddings]
    sparsity_after = [vis.compute_sparsity(emb) for emb in separated_embeddings]

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=sparsity_before,
                name="Before Separation",
                opacity=0.7,
                nbinsx=30,
                marker_color="blue"
            ))

            fig.add_trace(go.Histogram(
                x=sparsity_after,
                name="After Separation",
                opacity=0.7,
                nbinsx=30,
                marker_color="red"
            ))

            fig.update_layout(
                title="Sparsity Distribution",
                xaxis_title="Sparsity (fraction of near-zero values)",
                yaxis_title="Count",
                barmode="overlay",
                height=500
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

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(sparsity_before, bins=30, alpha=0.7, label="Before Separation",
                   color="blue", edgecolor="black")
            ax.hist(sparsity_after, bins=30, alpha=0.7, label="After Separation",
                   color="red", edgecolor="black")

            ax.set_xlabel("Sparsity (fraction of near-zero values)")
            ax.set_ylabel("Count")
            ax.set_title("Sparsity Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics
            mean_before = np.mean(sparsity_before)
            mean_after = np.mean(sparsity_after)
            ax.axvline(mean_before, color="blue", linestyle="--", linewidth=2,
                      label=f"Mean before: {mean_before:.3f}")
            ax.axvline(mean_after, color="red", linestyle="--", linewidth=2,
                      label=f"Mean after: {mean_after:.3f}")

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
    "PatternSeparationVisualizer",
    "plot_separation_comparison",
    "plot_sparsity_distribution",
]
