"""
Consolidation replay visualization for World Weaver.

Visualizes sleep-based memory consolidation:
- Sharp-wave ripple sequences
- Replay priority distributions
- NREM/REM phase dynamics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReplaySequence:
    """A sequence of memories replayed during SWR."""

    sequence_id: int
    memory_ids: list[str]
    replay_times: list[datetime]
    priority_scores: list[float]
    phase: str  # "nrem" or "rem"


class ConsolidationVisualizer:
    """
    Tracks and visualizes memory consolidation dynamics.

    Records replay sequences and consolidation events for analysis
    of sleep-based learning.
    """

    def __init__(self):
        """Initialize consolidation visualizer."""
        self._sequences: list[ReplaySequence] = []
        self._sequence_counter = 0

    def record_replay_sequence(
        self,
        memory_ids: list[str],
        priority_scores: list[float],
        phase: str = "nrem"
    ) -> None:
        """
        Record a replay sequence.

        Args:
            memory_ids: List of memory IDs in replay order
            priority_scores: Priority scores for each memory
            phase: Sleep phase ("nrem" or "rem")
        """
        now = datetime.now()
        replay_times = [now] * len(memory_ids)

        sequence = ReplaySequence(
            sequence_id=self._sequence_counter,
            memory_ids=memory_ids,
            replay_times=replay_times,
            priority_scores=priority_scores,
            phase=phase
        )

        self._sequences.append(sequence)
        self._sequence_counter += 1

    def get_priority_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get distribution of replay priorities.

        Returns:
            Tuple of (nrem_priorities, rem_priorities)
        """
        nrem_priorities = []
        rem_priorities = []

        for seq in self._sequences:
            if seq.phase == "nrem":
                nrem_priorities.extend(seq.priority_scores)
            elif seq.phase == "rem":
                rem_priorities.extend(seq.priority_scores)

        return np.array(nrem_priorities), np.array(rem_priorities)

    def get_sequence_lengths(self) -> tuple[list[int], list[int]]:
        """
        Get distribution of sequence lengths.

        Returns:
            Tuple of (nrem_lengths, rem_lengths)
        """
        nrem_lengths = []
        rem_lengths = []

        for seq in self._sequences:
            if seq.phase == "nrem":
                nrem_lengths.append(len(seq.memory_ids))
            elif seq.phase == "rem":
                rem_lengths.append(len(seq.memory_ids))

        return nrem_lengths, rem_lengths

    def get_replay_matrix(self) -> tuple[np.ndarray, list[str], list[int]]:
        """
        Get replay matrix for heatmap visualization.

        Returns:
            Tuple of (matrix, memory_ids, sequence_ids)
            Matrix[i, j] = 1 if memory j was in sequence i
        """
        if not self._sequences:
            return np.array([[]]), [], []

        # Collect all unique memory IDs
        all_mem_ids = set()
        for seq in self._sequences:
            all_mem_ids.update(seq.memory_ids)

        all_mem_ids = list(all_mem_ids)

        # Build binary matrix
        matrix = np.zeros((len(self._sequences), len(all_mem_ids)))

        for i, seq in enumerate(self._sequences):
            for mem_id in seq.memory_ids:
                if mem_id in all_mem_ids:
                    j = all_mem_ids.index(mem_id)
                    matrix[i, j] = 1.0

        sequence_ids = [seq.sequence_id for seq in self._sequences]

        return matrix, all_mem_ids, sequence_ids


def plot_swr_sequence(
    visualizer: ConsolidationVisualizer,
    sequence_index: int = 0,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot a single SWR replay sequence.

    Args:
        visualizer: ConsolidationVisualizer instance
        sequence_index: Index of sequence to plot
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    if sequence_index >= len(visualizer._sequences):
        logger.warning(f"Sequence index {sequence_index} out of range")
        return

    seq = visualizer._sequences[sequence_index]

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Plot memory activations over time
            for i, (mem_id, priority) in enumerate(zip(seq.memory_ids, seq.priority_scores)):
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[priority],
                    mode="markers+text",
                    marker=dict(size=15, color=priority, colorscale="Viridis",
                              showscale=True),
                    text=mem_id[:8],
                    textposition="top center",
                    name=mem_id[:8]
                ))

            # Connect with lines
            fig.add_trace(go.Scatter(
                x=list(range(len(seq.memory_ids))),
                y=seq.priority_scores,
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                showlegend=False
            ))

            fig.update_layout(
                title=f"SWR Replay Sequence {seq.sequence_id} ({seq.phase.upper()} phase)",
                xaxis_title="Position in Sequence",
                yaxis_title="Priority Score",
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

            fig, ax = plt.subplots(figsize=(12, 6))

            positions = list(range(len(seq.memory_ids)))

            # Plot points
            scatter = ax.scatter(
                positions,
                seq.priority_scores,
                c=seq.priority_scores,
                cmap="viridis",
                s=150,
                edgecolors="black",
                linewidths=1.5
            )

            # Connect with lines
            ax.plot(positions, seq.priority_scores, "gray", linestyle="--",
                   linewidth=1, alpha=0.5)

            # Add labels
            for i, mem_id in enumerate(seq.memory_ids):
                ax.text(i, seq.priority_scores[i] + 0.05, mem_id[:8],
                       ha="center", va="bottom", fontsize=8)

            ax.set_xlabel("Position in Sequence")
            ax.set_ylabel("Priority Score")
            ax.set_title(f"SWR Replay Sequence {seq.sequence_id} ({seq.phase.upper()} phase)")
            ax.grid(True, alpha=0.3)

            plt.colorbar(scatter, ax=ax, label="Priority")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


def plot_replay_priority(
    visualizer: ConsolidationVisualizer,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot distribution of replay priorities by phase.

    Args:
        visualizer: ConsolidationVisualizer instance
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    nrem_priorities, rem_priorities = visualizer.get_priority_distribution()

    if len(nrem_priorities) == 0 and len(rem_priorities) == 0:
        logger.warning("No replay data to plot")
        return

    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("NREM Replay Priorities", "REM Replay Priorities")
            )

            # NREM histogram
            if len(nrem_priorities) > 0:
                fig.add_trace(
                    go.Histogram(x=nrem_priorities, name="NREM",
                               marker_color="blue", nbinsx=30),
                    row=1, col=1
                )

            # REM histogram
            if len(rem_priorities) > 0:
                fig.add_trace(
                    go.Histogram(x=rem_priorities, name="REM",
                               marker_color="green", nbinsx=30),
                    row=1, col=2
                )

            fig.update_xaxes(title_text="Priority Score", row=1, col=1)
            fig.update_xaxes(title_text="Priority Score", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)

            fig.update_layout(
                title="Replay Priority Distributions",
                showlegend=False,
                height=400
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # NREM histogram
            if len(nrem_priorities) > 0:
                ax1.hist(nrem_priorities, bins=30, color="blue", alpha=0.7,
                        edgecolor="black")
                ax1.set_xlabel("Priority Score")
                ax1.set_ylabel("Count")
                ax1.set_title(f"NREM Replay Priorities (n={len(nrem_priorities)})")
                ax1.grid(True, alpha=0.3)

                # Add mean line
                mean_nrem = np.mean(nrem_priorities)
                ax1.axvline(mean_nrem, color="red", linestyle="--", linewidth=2,
                           label=f"Mean: {mean_nrem:.3f}")
                ax1.legend()

            # REM histogram
            if len(rem_priorities) > 0:
                ax2.hist(rem_priorities, bins=30, color="green", alpha=0.7,
                        edgecolor="black")
                ax2.set_xlabel("Priority Score")
                ax2.set_ylabel("Count")
                ax2.set_title(f"REM Replay Priorities (n={len(rem_priorities)})")
                ax2.grid(True, alpha=0.3)

                # Add mean line
                mean_rem = np.mean(rem_priorities)
                ax2.axvline(mean_rem, color="red", linestyle="--", linewidth=2,
                           label=f"Mean: {mean_rem:.3f}")
                ax2.legend()

            fig.suptitle("Replay Priority Distributions")
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
    "ConsolidationVisualizer",
    "ReplaySequence",
    "plot_replay_priority",
    "plot_swr_sequence",
]
