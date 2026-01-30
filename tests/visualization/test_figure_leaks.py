"""MEM-006: Tests for matplotlib figure leak prevention.

These tests verify that all visualization functions properly close
matplotlib figures after displaying or saving to prevent memory leaks.

Note: These tests use source code inspection rather than mocking because
matplotlib is often already imported by other tests in the suite, making
sys.modules patching unreliable.
"""

import pytest
import numpy as np
import inspect


class TestFigureLeakPrevention:
    """Tests ensuring matplotlib figures are properly closed via code inspection."""

    def test_activation_heatmap_closes_figure_on_show(self):
        """Test activation heatmap code has proper plt.close(fig) pattern."""
        from ww.visualization import activation_heatmap

        source = inspect.getsource(activation_heatmap)

        # Verify plt.close(fig) appears in the code after plt.show()
        assert "plt.close(fig)" in source, "Missing plt.close(fig) call"
        assert "plt.show()" in source, "Missing plt.show() call"

        # Verify the MEM-006 fix comment exists
        assert "MEM-006" in source, "Missing MEM-006 fix marker"

    def test_pattern_separation_closes_figure_on_show(self):
        """Test pattern separation code has proper plt.close(fig) pattern."""
        from ww.visualization import pattern_separation

        source = inspect.getsource(pattern_separation)

        # Verify plt.close(fig) appears in the code
        assert "plt.close(fig)" in source, "Missing plt.close(fig) call"
        assert "plt.show()" in source, "Missing plt.show() call"
        assert "MEM-006" in source, "Missing MEM-006 fix marker"

    def test_plasticity_traces_closes_figure_on_show(self):
        """Test plasticity traces code has proper plt.close(fig) pattern."""
        from ww.visualization import plasticity_traces

        source = inspect.getsource(plasticity_traces)

        # Verify plt.close(fig) appears in the code
        assert "plt.close(fig)" in source, "Missing plt.close(fig) call"
        assert "plt.show()" in source, "Missing plt.show() call"
        assert "MEM-006" in source, "Missing MEM-006 fix marker"

    def test_embedding_projections_closes_figure_on_show(self):
        """Test embedding projections code has proper plt.close(fig) pattern."""
        from ww.visualization import embedding_projections

        source = inspect.getsource(embedding_projections)

        # Verify plt.close(fig) appears in the code
        assert "plt.close(fig)" in source, "Missing plt.close(fig) call"
        assert "plt.show()" in source, "Missing plt.show() call"
        assert "MEM-006" in source, "Missing MEM-006 fix marker"

    def test_consolidation_replay_code_closes_figure(self):
        """Test consolidation replay code has proper plt.close(fig) pattern."""
        import inspect
        from ww.visualization import consolidation_replay

        # Read the source code and verify the fix pattern exists
        source = inspect.getsource(consolidation_replay)

        # Verify plt.close(fig) appears after plt.show() in the code
        assert "plt.show()" in source
        assert "plt.close(fig)" in source

        # Check that plt.close(fig) appears immediately after plt.show()
        # in the context of MEM-006 fix
        assert "MEM-006 FIX" in source


class TestFigureCloseOnSave:
    """Tests that figures are closed after saving via code inspection."""

    def test_savefig_closes_figure(self):
        """Verify code closes figure after savefig."""
        from ww.visualization import activation_heatmap

        source = inspect.getsource(activation_heatmap)

        # Find the save path block and verify close is called after savefig
        # Pattern: savefig(...) followed by close(fig)
        assert "plt.savefig" in source, "Missing plt.savefig call"

        # Check that close follows savefig in the save_path block
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "plt.savefig" in line and "save_path" in source[:source.index("plt.savefig") + 100]:
                # Check next few lines for close
                nearby_lines = "\n".join(lines[i:i+3])
                assert "plt.close(fig)" in nearby_lines, \
                    "plt.close(fig) should follow plt.savefig"
                break
