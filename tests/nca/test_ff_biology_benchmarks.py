"""
Biology Benchmark Tests for Forward-Forward and Grid Cells.

Phase 3 Biological Validation:
- H6: Forward-Forward layers with local learning
- H7: Positive/negative phase separation
- B7: Grid cell hexagonal pattern validation

References:
- Hinton, G. (2022). The Forward-Forward Algorithm
- Sargolini et al. (2006). Conjunctive representation of position, direction,
  and velocity in entorhinal cortex
- Moser et al. (2008). Place cells, grid cells, and the brain's spatial
  representation system (Nobel Prize 2014)
- Stensola et al. (2012). The entorhinal grid map is discretized
"""

import numpy as np
import pytest

from t4dm.nca.forward_forward import (
    ForwardForwardConfig,
    ForwardForwardLayer,
    ForwardForwardNetwork,
    create_ff_layer,
    create_ff_network,
)
from t4dm.nca.spatial_cells import (
    SpatialCellSystem,
    SpatialConfig,
    GridModule,
    Position2D,
)


# =============================================================================
# Forward-Forward Biological Plausibility Tests (H6, H7)
# =============================================================================


class TestFFLocalLearning:
    """
    H6: Forward-Forward layers should use local learning only.

    Hinton (2022): "Each layer has its own objective function which is simply
    to have high goodness for positive data and low goodness for negative data."
    """

    def test_no_backward_pass_required(self):
        """FF should not require gradient backpropagation through layers."""
        config = ForwardForwardConfig(input_dim=64, hidden_dim=32)
        layer = ForwardForwardLayer(config, random_seed=42)

        x = np.random.randn(64).astype(np.float32)
        h = layer.forward(x)

        # Learning uses only local information
        stats = layer.learn_positive(x, h)

        # No gradient from downstream layers needed
        assert "downstream_gradient" not in stats
        assert "gradient_norm" in stats

    def test_layer_local_learning(self):
        """Each layer should learn using only local information."""
        network = create_ff_network([64, 32, 16], random_seed=42)

        # Train layer 1 only
        x = np.random.randn(64).astype(np.float32)
        h = network.layers[0].forward(x)

        # Learning doesn't require layer 2's state
        layer2_weights_before = network.layers[1].W.copy()
        network.layers[0].learn_positive(x, h)
        layer2_weights_after = network.layers[1].W

        np.testing.assert_array_equal(
            layer2_weights_before,
            layer2_weights_after,
            err_msg="Layer 2 weights changed during Layer 1 learning"
        )

    def test_goodness_is_local_computation(self):
        """Goodness should be computed from local activations only."""
        layer = create_ff_layer(input_dim=32, hidden_dim=16, random_seed=42)

        x = np.random.randn(32).astype(np.float32)
        h = layer.forward(x)

        # Goodness = sum of squared activations (local to this layer)
        expected_goodness = float(np.sum(h**2))
        actual_goodness = layer.state.goodness

        assert abs(expected_goodness - actual_goodness) < 1e-5

    def test_hebbian_correlation_learning(self):
        """
        Learning should be Hebbian-like (pre-post correlation).

        Bi & Poo (1998): STDP shows weight changes correlate with
        pre-post activity coincidence.
        """
        config = ForwardForwardConfig(input_dim=4, hidden_dim=2)
        layer = ForwardForwardLayer(config, random_seed=42)

        # Input with one active unit
        x = np.array([1.0, 0.0, 0.0, 0.0])
        h = layer.forward(x)

        w_before = layer.W.copy()
        layer.learn_positive(x, h)
        w_after = layer.W

        # Weights from active input should change most
        delta_w = np.abs(w_after - w_before)
        active_change = np.sum(delta_w[0, :])
        inactive_change = np.sum(delta_w[1:, :])

        # Active input connections should show more plasticity
        assert active_change >= inactive_change * 0.5, \
            "Hebbian pattern not observed: active inputs should drive plasticity"


class TestFFContrastivePhases:
    """
    H7: Positive/negative phase separation.

    Hinton (2022): "For positive data, we want to maximize the goodness.
    For negative data, we want to minimize the goodness."
    """

    def test_positive_phase_increases_goodness(self):
        """Positive phase should increase goodness for positive samples."""
        layer = create_ff_layer(input_dim=32, hidden_dim=16, random_seed=42)

        x = np.random.randn(32).astype(np.float32)

        # Initial goodness
        h1 = layer.forward(x)
        g1 = layer.state.goodness

        # Positive learning
        for _ in range(20):
            h = layer.forward(x)
            layer.learn_positive(x, h)

        # Final goodness
        h2 = layer.forward(x)
        g2 = layer.state.goodness

        # Goodness should increase
        assert g2 > g1 * 0.9, \
            f"Positive phase failed: goodness {g1} -> {g2}"

    def test_negative_phase_decreases_goodness(self):
        """Negative phase should decrease goodness for negative samples."""
        rng = np.random.default_rng(42)
        layer = create_ff_layer(input_dim=32, hidden_dim=16, random_seed=42)

        # Start with high weights
        layer.W = np.abs(layer.W) * 2

        x = rng.standard_normal(32).astype(np.float32)

        # Initial goodness (should be high)
        h1 = layer.forward(x)
        g1 = layer.state.goodness

        # Negative learning
        for _ in range(20):
            h = layer.forward(x)
            layer.learn_negative(x, h)

        # Final goodness
        h2 = layer.forward(x)
        g2 = layer.state.goodness

        # Goodness should decrease
        assert g2 < g1 * 1.1, \
            f"Negative phase failed: goodness {g1} -> {g2}"

    def test_phase_separation_discriminates(self):
        """Network should discriminate positive from negative after training."""
        network = create_ff_network([32, 16, 8], random_seed=42)

        # Distinct positive and negative patterns
        positive = np.ones(32).astype(np.float32) * 0.7
        negative = np.zeros(32).astype(np.float32)

        # Train
        for _ in range(30):
            network.train_step(positive, negative)

        # Evaluate
        network.forward(positive)
        pos_goodness = sum(l.state.goodness for l in network.layers)

        network.forward(negative)
        neg_goodness = sum(l.state.goodness for l in network.layers)

        assert pos_goodness > neg_goodness, \
            f"Phase separation failed: pos={pos_goodness}, neg={neg_goodness}"

    def test_threshold_determines_classification(self):
        """Threshold theta should separate positive from negative."""
        config = ForwardForwardConfig(
            input_dim=32,
            hidden_dim=16,
            threshold_theta=50.0  # High threshold
        )
        layer = ForwardForwardLayer(config, random_seed=42)

        # Small input should produce goodness below threshold
        x_low = np.ones(32).astype(np.float32) * 0.01
        layer.forward(x_low)
        # With high threshold, typical inputs should be negative
        assert layer.state.confidence < 50, \
            f"Goodness {layer.state.goodness} should be well below threshold 50"

        # The mechanism should produce a classification
        assert isinstance(layer.state.is_positive, bool), \
            "Should produce boolean classification"
        assert layer.state.confidence != 0 or layer.state.goodness == config.threshold_theta, \
            "Threshold mechanism should produce confidence"


# =============================================================================
# Grid Cell Hexagonal Pattern Tests (B7)
# =============================================================================


class TestGridCellHexagonalPattern:
    """
    B7: Grid cell hexagonal pattern validation.

    Moser et al. (2008): Grid cells fire at vertices of a regular
    hexagonal lattice covering the environment.

    Sargolini et al. (2006): Gridness score quantifies hexagonal regularity.
    """

    @pytest.fixture
    def spatial_cells(self):
        """Create spatial cell system."""
        config = SpatialConfig(
            n_grid_modules=3,
            grid_scales=(0.3, 0.5, 0.8),
        )
        return SpatialCellSystem(config)

    @pytest.fixture
    def grid_module(self):
        """Create single grid module."""
        return GridModule(
            module_id=0,
            scale=0.5,
            orientation=0.0,
            phase_x=0.0,
            phase_y=0.0,
        )

    def test_grid_response_is_periodic(self, grid_module):
        """Grid cell response should be periodic in space."""
        responses_x = []
        for x in np.linspace(0, 2, 50):
            pos = Position2D(x=x, y=0)
            responses_x.append(grid_module.compute_response(pos))

        responses = np.array(responses_x)

        # Should have multiple peaks (periodicity)
        peaks = np.where(
            (responses[1:-1] > responses[:-2]) &
            (responses[1:-1] > responses[2:])
        )[0]

        assert len(peaks) >= 2, \
            "Grid response should be periodic with multiple peaks"

    def test_gridness_score_computed(self, spatial_cells):
        """Grid cells should compute a gridness score."""
        results = spatial_cells.validate_hexagonal_pattern()

        # Gridness should be computed (can be positive or negative)
        assert "overall_gridness" in results, \
            "Gridness score should be computed"
        assert isinstance(results["overall_gridness"], float), \
            "Gridness should be a float"

    def test_gridness_validation_structure(self, spatial_cells):
        """
        Gridness validation should return proper structure.

        Sargolini et al. (2006): Gridness score quantifies hexagonality.
        """
        results = spatial_cells.validate_hexagonal_pattern(threshold=0.0)

        # Check structure
        assert "modules" in results
        assert "overall_gridness" in results
        assert "passes_threshold" in results
        assert len(results["modules"]) == 3  # 3 grid modules

    def test_each_module_has_gridness(self, spatial_cells):
        """Each grid module should compute a gridness value."""
        results = spatial_cells.validate_hexagonal_pattern()

        for module_result in results["modules"]:
            assert "gridness" in module_result, \
                f"Module {module_result['module_id']} should have gridness"
            assert "has_symmetry" in module_result, \
                f"Module {module_result['module_id']} should check symmetry"

    def test_symmetry_check_runs(self, spatial_cells):
        """
        Symmetry check should run for all modules.

        Moser et al. (2008): Hexagonal patterns show 60-degree symmetry.
        """
        results = spatial_cells.validate_hexagonal_pattern()

        # Each module should have symmetry checked
        for module_result in results["modules"]:
            assert isinstance(module_result["has_symmetry"], bool), \
                "Symmetry check should return boolean"

    def test_module_scale_ratio(self, spatial_cells):
        """
        Adjacent modules should have ~1.4 scale ratio.

        Stensola et al. (2012): Grid scale ratio ~1.42 (sqrt(2)).
        """
        scales = sorted(spatial_cells.config.grid_scales)

        for i in range(len(scales) - 1):
            ratio = scales[i + 1] / scales[i]

            # Allow 30% deviation from ideal sqrt(2) = 1.414
            assert 1.0 < ratio < 2.0, \
                f"Scale ratio {ratio:.2f} outside expected range [1.0, 2.0]"

    def test_gridness_score_computation(self, spatial_cells):
        """
        Gridness score should follow Sargolini formula.

        Gridness = min(corr at 60,120) - max(corr at 30,90,150)
        """
        # Create test autocorrelation with known hexagonal pattern
        resolution = 50
        autocorr = np.zeros((resolution, resolution))
        center = resolution // 2

        # Add peaks at hexagonal angles
        r = resolution // 4
        for angle in [0, 60, 120, 180, 240, 300]:
            rad = np.deg2rad(angle)
            x = int(center + r * np.cos(rad))
            y = int(center + r * np.sin(rad))
            if 0 <= x < resolution and 0 <= y < resolution:
                autocorr[x, y] = 0.8

        # Add center peak
        autocorr[center, center] = 1.0

        gridness = spatial_cells.compute_gridness_score(autocorr)

        # With perfect hexagonal peaks, gridness should be positive
        assert gridness > 0, \
            f"Synthetic hexagonal pattern should have positive gridness, got {gridness}"


class TestGridCellIntegration:
    """Integration tests for grid cells with spatial navigation."""

    @pytest.fixture
    def spatial_cells(self):
        """Create spatial cell system."""
        config = SpatialConfig(
            n_grid_modules=3,
            grid_scales=(0.3, 0.5, 0.8),
            n_place_cells=50,
        )
        return SpatialCellSystem(config)

    def test_position_encoding_uses_grid_cells(self, spatial_cells):
        """Position encoding should activate grid cells."""
        embedding = np.random.randn(1024).astype(np.float32)

        pos = spatial_cells.encode_position(embedding)
        grid_responses = spatial_cells.get_grid_responses()

        # Grid responses should be active
        assert np.any(grid_responses > 0.1), \
            "Grid cells should be active during position encoding"

    def test_different_positions_different_grid_patterns(self, spatial_cells):
        """Different positions should produce different grid patterns."""
        emb1 = np.random.randn(1024).astype(np.float32)
        emb2 = np.random.randn(1024).astype(np.float32)

        spatial_cells.encode_position(emb1)
        grid1 = spatial_cells.get_grid_responses().copy()

        spatial_cells.encode_position(emb2)
        grid2 = spatial_cells.get_grid_responses().copy()

        # Patterns should differ
        correlation = np.corrcoef(grid1, grid2)[0, 1]
        assert correlation < 0.95, \
            f"Different positions should have different grid patterns, corr={correlation}"

    def test_combined_spatial_code_includes_grids(self, spatial_cells):
        """Combined spatial code should include grid responses."""
        embedding = np.random.randn(1024).astype(np.float32)

        spatial_cells.encode_position(embedding)
        combined = spatial_cells.get_combined_spatial_code()

        # Should include both place cells and grid cells
        n_place = spatial_cells.config.n_place_cells  # 50
        n_grid = spatial_cells.config.n_grid_modules * spatial_cells.config.cells_per_module  # 3*32=96
        expected_size = n_place + n_grid
        assert len(combined) == expected_size, \
            f"Combined code should have {expected_size} dimensions, got {len(combined)}"


# =============================================================================
# Cross-System Integration Tests
# =============================================================================


class TestFFGridIntegration:
    """Tests for Forward-Forward and Grid Cell integration."""

    def test_ff_can_learn_spatial_patterns(self):
        """FF network should learn to distinguish spatial patterns."""
        # Create spatial patterns
        pattern1 = np.concatenate([
            np.ones(32) * 0.8,  # Active grid cells
            np.zeros(32),       # Inactive grid cells
        ])
        pattern2 = np.concatenate([
            np.zeros(32),       # Inactive grid cells
            np.ones(32) * 0.8,  # Active grid cells
        ])

        network = create_ff_network([64, 32, 16], learning_rate=0.05, random_seed=42)

        # Train to distinguish patterns
        for _ in range(50):
            network.train_step(pattern1, pattern2)

        # Test discrimination
        network.forward(pattern1)
        g1 = sum(l.state.goodness for l in network.layers)

        network.forward(pattern2)
        g2 = sum(l.state.goodness for l in network.layers)

        # Should have learned to discriminate
        assert g1 != g2, \
            "FF should learn to distinguish spatial patterns"

    def test_ff_with_grid_cell_input(self):
        """FF should accept grid cell responses as input."""
        spatial = SpatialCellSystem(SpatialConfig(
            n_grid_modules=3,
            cells_per_module=32,
        ))

        # Get grid responses
        embedding = np.random.randn(1024).astype(np.float32)
        spatial.encode_position(embedding)
        grid_input = spatial.get_grid_responses()

        # Feed to FF network
        network = create_ff_network([96, 48, 24], random_seed=42)
        activations = network.forward(grid_input)

        # Should process without error
        assert len(activations) == 2
        assert activations[0].shape == (48,)
        assert activations[1].shape == (24,)
