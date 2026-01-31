"""Tests for consolidation replay visualization module."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from t4dm.visualization.consolidation_replay import (
    ConsolidationVisualizer,
    ReplaySequence,
    plot_swr_sequence,
    plot_replay_priority,
)


class TestReplaySequence:
    """Tests for ReplaySequence dataclass."""

    def test_create_sequence(self):
        """Test creating a replay sequence."""
        sequence = ReplaySequence(
            sequence_id=1,
            memory_ids=["mem1", "mem2", "mem3"],
            replay_times=[datetime.now()] * 3,
            priority_scores=[0.8, 0.6, 0.7],
            phase="nrem",
        )
        assert sequence.sequence_id == 1
        assert len(sequence.memory_ids) == 3
        assert sequence.phase == "nrem"

    def test_sequence_fields(self):
        """Test sequence field access."""
        now = datetime.now()
        sequence = ReplaySequence(
            sequence_id=42,
            memory_ids=["a", "b"],
            replay_times=[now, now],
            priority_scores=[0.5, 0.9],
            phase="rem",
        )
        assert sequence.sequence_id == 42
        assert sequence.priority_scores[1] == 0.9


class TestConsolidationVisualizer:
    """Tests for ConsolidationVisualizer class."""

    def test_init(self):
        """Test initialization."""
        viz = ConsolidationVisualizer()
        assert len(viz._sequences) == 0
        assert viz._sequence_counter == 0

    def test_record_replay_sequence(self):
        """Test recording a replay sequence."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(
            memory_ids=["ep1", "ep2", "ep3"],
            priority_scores=[0.8, 0.6, 0.7],
            phase="nrem",
        )
        assert len(viz._sequences) == 1
        assert viz._sequences[0].sequence_id == 0
        assert len(viz._sequences[0].memory_ids) == 3

    def test_record_multiple_sequences(self):
        """Test recording multiple replay sequences."""
        viz = ConsolidationVisualizer()
        for i in range(5):
            viz.record_replay_sequence(
                memory_ids=[f"ep{i}", f"ep{i+1}"],
                priority_scores=[0.5, 0.7],
                phase="nrem" if i % 2 == 0 else "rem",
            )
        assert len(viz._sequences) == 5
        assert viz._sequence_counter == 5

    def test_get_priority_distribution(self):
        """Test getting priority distribution by phase."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(["m1", "m2"], [0.8, 0.6], "nrem")
        viz.record_replay_sequence(["m3", "m4"], [0.9, 0.7], "nrem")
        viz.record_replay_sequence(["m5"], [0.5], "rem")

        nrem_priorities, rem_priorities = viz.get_priority_distribution()
        assert len(nrem_priorities) == 4  # 2 + 2 from nrem sequences
        assert len(rem_priorities) == 1   # 1 from rem sequence

    def test_get_priority_distribution_empty(self):
        """Test priority distribution with no data."""
        viz = ConsolidationVisualizer()
        nrem_priorities, rem_priorities = viz.get_priority_distribution()
        assert len(nrem_priorities) == 0
        assert len(rem_priorities) == 0

    def test_get_sequence_lengths(self):
        """Test getting sequence lengths by phase."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(["m1", "m2", "m3"], [0.5]*3, "nrem")
        viz.record_replay_sequence(["m4", "m5"], [0.5]*2, "nrem")
        viz.record_replay_sequence(["m6"], [0.5], "rem")

        nrem_lengths, rem_lengths = viz.get_sequence_lengths()
        assert nrem_lengths == [3, 2]
        assert rem_lengths == [1]

    def test_get_replay_matrix(self):
        """Test getting replay matrix."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(["m1", "m2"], [0.5, 0.6], "nrem")
        viz.record_replay_sequence(["m2", "m3"], [0.7, 0.8], "nrem")

        matrix, mem_ids, seq_ids = viz.get_replay_matrix()
        assert matrix.shape[0] == 2  # 2 sequences
        assert len(seq_ids) == 2
        # m2 should appear in both sequences
        if "m2" in mem_ids:
            m2_idx = mem_ids.index("m2")
            assert matrix[0, m2_idx] == 1.0
            assert matrix[1, m2_idx] == 1.0

    def test_get_replay_matrix_empty(self):
        """Test replay matrix with no data."""
        viz = ConsolidationVisualizer()
        matrix, mem_ids, seq_ids = viz.get_replay_matrix()
        assert matrix.size == 0
        assert len(mem_ids) == 0
        assert len(seq_ids) == 0


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

    def test_data_for_swr_sequence(self):
        """Test data preparation for SWR sequence (no actual plotting)."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(
            memory_ids=["m1", "m2", "m3", "m4"],
            priority_scores=[0.8, 0.6, 0.9, 0.5],
            phase="nrem",
        )
        # Verify sequence data is retrievable
        assert len(viz._sequences) == 1
        assert viz._sequences[0].memory_ids == ["m1", "m2", "m3", "m4"]

    def test_data_for_replay_priority(self):
        """Test data preparation for replay priority distribution."""
        viz = ConsolidationVisualizer()
        for i in range(5):
            viz.record_replay_sequence(
                memory_ids=[f"m{i}"],
                priority_scores=[0.5 + 0.1*i],
                phase="nrem" if i % 2 == 0 else "rem",
            )
        # Verify priority distribution data is available
        nrem_priorities, rem_priorities = viz.get_priority_distribution()
        assert len(nrem_priorities) == 3  # 3 nrem sequences
        assert len(rem_priorities) == 2   # 2 rem sequences

    def test_empty_visualizer_data(self):
        """Test data from empty visualizer."""
        viz = ConsolidationVisualizer()
        nrem_priorities, rem_priorities = viz.get_priority_distribution()
        assert len(nrem_priorities) == 0
        assert len(rem_priorities) == 0
        matrix, mem_ids, seq_ids = viz.get_replay_matrix()
        assert matrix.size == 0


class TestConsolidationVisualizerAnalysis:
    """Extended tests for ConsolidationVisualizer analysis methods."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with mixed NREM/REM sequences."""
        viz = ConsolidationVisualizer()
        # NREM sequences
        viz.record_replay_sequence(
            memory_ids=["ep1", "ep2", "ep3", "ep4"],
            priority_scores=[0.9, 0.7, 0.8, 0.6],
            phase="nrem"
        )
        viz.record_replay_sequence(
            memory_ids=["ep2", "ep5"],
            priority_scores=[0.85, 0.75],
            phase="nrem"
        )
        # REM sequences
        viz.record_replay_sequence(
            memory_ids=["ep1", "ep3", "ep6"],
            priority_scores=[0.65, 0.55, 0.45],
            phase="rem"
        )
        viz.record_replay_sequence(
            memory_ids=["ep7"],
            priority_scores=[0.95],
            phase="rem"
        )
        return viz

    def test_priority_distribution_values(self, populated_visualizer):
        """Test priority distribution contains correct values."""
        nrem, rem = populated_visualizer.get_priority_distribution()

        # NREM should have 4 + 2 = 6 priorities
        assert len(nrem) == 6
        assert 0.9 in nrem
        assert 0.85 in nrem

        # REM should have 3 + 1 = 4 priorities
        assert len(rem) == 4
        assert 0.95 in rem
        assert 0.65 in rem

    def test_sequence_lengths_detailed(self, populated_visualizer):
        """Test sequence lengths are correct."""
        nrem_lengths, rem_lengths = populated_visualizer.get_sequence_lengths()

        assert nrem_lengths == [4, 2]  # 2 NREM sequences
        assert rem_lengths == [3, 1]   # 2 REM sequences

    def test_replay_matrix_structure(self, populated_visualizer):
        """Test replay matrix has correct structure."""
        matrix, mem_ids, seq_ids = populated_visualizer.get_replay_matrix()

        # 4 sequences total
        assert matrix.shape[0] == 4
        assert len(seq_ids) == 4

        # Should have 7 unique memories (ep1-ep7)
        assert len(mem_ids) == 7

        # All values should be 0 or 1
        assert np.all((matrix == 0) | (matrix == 1))

    def test_replay_matrix_overlap(self, populated_visualizer):
        """Test replay matrix captures memory overlap."""
        matrix, mem_ids, seq_ids = populated_visualizer.get_replay_matrix()

        # ep1 appears in sequence 0 and 2
        if "ep1" in mem_ids:
            ep1_idx = mem_ids.index("ep1")
            assert matrix[0, ep1_idx] == 1.0
            assert matrix[2, ep1_idx] == 1.0
            assert matrix[1, ep1_idx] == 0.0  # Not in sequence 1

    def test_sequence_counter_increments(self):
        """Test sequence counter increments correctly."""
        viz = ConsolidationVisualizer()

        for i in range(10):
            viz.record_replay_sequence(
                memory_ids=[f"m{i}"],
                priority_scores=[0.5],
                phase="nrem"
            )
            assert viz._sequence_counter == i + 1
            assert viz._sequences[i].sequence_id == i


class TestPlotSWRSequence:
    """Tests for plot_swr_sequence function."""

    @pytest.fixture
    def visualizer_with_sequences(self):
        """Create visualizer with multiple sequences."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(
            memory_ids=["memory_a", "memory_b", "memory_c"],
            priority_scores=[0.9, 0.7, 0.8],
            phase="nrem"
        )
        viz.record_replay_sequence(
            memory_ids=["memory_d", "memory_e"],
            priority_scores=[0.6, 0.5],
            phase="rem"
        )
        return viz

    def test_plot_swr_sequence_matplotlib(self, visualizer_with_sequences):
        """Test SWR sequence plot with matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        # Should not raise - plot first sequence
        plot_swr_sequence(visualizer_with_sequences, sequence_index=0)
        plt.close("all")

    def test_plot_swr_sequence_second_sequence(self, visualizer_with_sequences):
        """Test plotting second sequence."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_swr_sequence(visualizer_with_sequences, sequence_index=1)
        plt.close("all")

    def test_plot_swr_sequence_out_of_range(self, visualizer_with_sequences):
        """Test plotting with out of range index logs warning."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        # Should log warning but not raise
        plot_swr_sequence(visualizer_with_sequences, sequence_index=99)
        plt.close("all")

    def test_plot_swr_sequence_save_path(self, visualizer_with_sequences, tmp_path):
        """Test saving SWR sequence plot to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        save_path = tmp_path / "swr_sequence.png"
        plot_swr_sequence(
            visualizer_with_sequences,
            sequence_index=0,
            save_path=save_path
        )
        assert save_path.exists()
        plt.close("all")


class TestPlotReplayPriority:
    """Tests for plot_replay_priority function."""

    @pytest.fixture
    def mixed_phase_visualizer(self):
        """Create visualizer with both NREM and REM data."""
        viz = ConsolidationVisualizer()
        # Multiple NREM sequences
        for i in range(5):
            viz.record_replay_sequence(
                memory_ids=[f"nrem_m{i}"],
                priority_scores=[0.3 + 0.1 * i],
                phase="nrem"
            )
        # Multiple REM sequences
        for i in range(3):
            viz.record_replay_sequence(
                memory_ids=[f"rem_m{i}"],
                priority_scores=[0.5 + 0.15 * i],
                phase="rem"
            )
        return viz

    def test_plot_replay_priority_matplotlib(self, mixed_phase_visualizer):
        """Test replay priority plot with matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_replay_priority(mixed_phase_visualizer)
        plt.close("all")

    def test_plot_replay_priority_save_path(self, mixed_phase_visualizer, tmp_path):
        """Test saving replay priority plot to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        save_path = tmp_path / "replay_priority.png"
        plot_replay_priority(mixed_phase_visualizer, save_path=save_path)
        assert save_path.exists()
        plt.close("all")

    def test_plot_replay_priority_empty(self):
        """Test plotting empty visualizer logs warning."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = ConsolidationVisualizer()
        # Should log warning but not raise
        plot_replay_priority(viz)
        plt.close("all")

    def test_plot_replay_priority_nrem_only(self):
        """Test plotting with only NREM data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(["m1", "m2"], [0.8, 0.6], "nrem")
        viz.record_replay_sequence(["m3"], [0.7], "nrem")

        plot_replay_priority(viz)
        plt.close("all")

    def test_plot_replay_priority_rem_only(self):
        """Test plotting with only REM data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(["m1", "m2"], [0.5, 0.4], "rem")
        viz.record_replay_sequence(["m3"], [0.6], "rem")

        plot_replay_priority(viz)
        plt.close("all")


class TestConsolidationReplayEdgeCases:
    """Edge case tests for consolidation replay visualization."""

    def test_single_memory_sequence(self):
        """Test sequence with single memory."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(
            memory_ids=["only_one"],
            priority_scores=[0.99],
            phase="nrem"
        )

        matrix, mem_ids, seq_ids = viz.get_replay_matrix()
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1.0

    def test_very_long_sequence(self):
        """Test very long replay sequence."""
        viz = ConsolidationVisualizer()
        n_memories = 100
        viz.record_replay_sequence(
            memory_ids=[f"m{i}" for i in range(n_memories)],
            priority_scores=[0.5 + 0.005 * i for i in range(n_memories)],
            phase="nrem"
        )

        nrem, rem = viz.get_priority_distribution()
        assert len(nrem) == n_memories
        assert len(rem) == 0

    def test_many_short_sequences(self):
        """Test many short sequences."""
        viz = ConsolidationVisualizer()
        for i in range(50):
            viz.record_replay_sequence(
                memory_ids=[f"m{i}"],
                priority_scores=[i / 50.0],
                phase="nrem" if i % 2 == 0 else "rem"
            )

        nrem, rem = viz.get_priority_distribution()
        assert len(nrem) == 25
        assert len(rem) == 25

    def test_overlapping_memories_across_phases(self):
        """Test same memory in both NREM and REM."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(["shared", "nrem_only"], [0.8, 0.7], "nrem")
        viz.record_replay_sequence(["shared", "rem_only"], [0.9, 0.6], "rem")

        matrix, mem_ids, seq_ids = viz.get_replay_matrix()

        # shared should appear in both
        shared_idx = mem_ids.index("shared")
        assert matrix[0, shared_idx] == 1.0
        assert matrix[1, shared_idx] == 1.0

    def test_zero_priority_scores(self):
        """Test sequences with zero priority scores."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(
            memory_ids=["m1", "m2"],
            priority_scores=[0.0, 0.0],
            phase="nrem"
        )

        nrem, rem = viz.get_priority_distribution()
        assert np.all(nrem == 0.0)

    def test_high_priority_scores(self):
        """Test sequences with very high priority scores."""
        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(
            memory_ids=["m1", "m2"],
            priority_scores=[1.0, 0.999],
            phase="rem"
        )

        nrem, rem = viz.get_priority_distribution()
        assert 1.0 in rem
        assert 0.999 in rem

    def test_sequence_with_long_memory_ids(self):
        """Test sequence with very long memory IDs."""
        viz = ConsolidationVisualizer()
        long_id = "a" * 1000
        viz.record_replay_sequence(
            memory_ids=[long_id, "short"],
            priority_scores=[0.5, 0.6],
            phase="nrem"
        )

        matrix, mem_ids, seq_ids = viz.get_replay_matrix()
        assert long_id in mem_ids

    def test_plot_long_sequence(self):
        """Test plotting sequence with many memories."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = ConsolidationVisualizer()
        viz.record_replay_sequence(
            memory_ids=[f"mem_{i:03d}" for i in range(20)],
            priority_scores=[0.5 + 0.02 * i for i in range(20)],
            phase="nrem"
        )

        plot_swr_sequence(viz, sequence_index=0)
        plt.close("all")
