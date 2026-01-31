"""
Unit tests for attractor network implementation.
"""

import pytest
import torch
import torch.nn.functional as F

from t4dm.encoding.attractor import (
    AttractorNetwork,
    ModernHopfieldNetwork,
    RetrievalResult
)


class TestAttractorNetwork:
    """Tests for Hopfield-style attractor network."""

    @pytest.fixture
    def network(self):
        """Create default attractor network."""
        return AttractorNetwork(
            dim=256,
            settling_steps=20,
            step_size=0.1,
            capacity_ratio=0.14
        )

    def test_initialization(self, network):
        """Network initializes correctly."""
        assert network.dim == 256
        assert network.capacity == int(256 * 0.14)
        assert network.pattern_count == 0

    def test_store_pattern(self, network):
        """Storing patterns works correctly."""
        pattern = torch.randn(256)

        result = network.store(pattern, "test_pattern")

        assert result["stored"] == True
        assert result["pattern_id"] == "test_pattern"
        assert network.pattern_count == 1

    def test_store_multiple_patterns(self, network):
        """Storing multiple patterns works."""
        for i in range(5):
            pattern = torch.randn(256)
            result = network.store(pattern, f"pattern_{i}")
            assert result["stored"] == True

        assert network.pattern_count == 5

    def test_capacity_limit(self, network):
        """Cannot exceed capacity."""
        # Fill to capacity
        for i in range(network.capacity):
            pattern = torch.randn(256)
            network.store(pattern, f"pattern_{i}")

        # Try to store one more
        extra = torch.randn(256)
        result = network.store(extra, "extra")

        assert result["stored"] == False
        assert "Capacity" in result["reason"]

    def test_retrieve_exact(self, network):
        """Exact cue retrieves correct pattern."""
        pattern = torch.randn(256)
        network.store(pattern, "test")

        result = network.retrieve(pattern)

        assert result.pattern_id == "test"
        assert result.confidence > 0.9

    def test_retrieve_noisy(self, network):
        """Noisy cue retrieves correct pattern."""
        pattern = torch.randn(256)
        network.store(pattern, "test")

        # Add noise
        noisy = pattern + 0.3 * torch.randn(256)

        result = network.retrieve(noisy)

        assert result.pattern_id == "test"
        assert result.confidence > 0.7

    def test_settling_dynamics(self, network):
        """Energy decreases during settling."""
        # Store some patterns
        for i in range(3):
            network.store(torch.randn(256), f"p{i}")

        # Retrieve with trajectory tracking
        cue = torch.randn(256)
        result = network.retrieve(cue, track_trajectory=True)

        assert result.trajectory is not None
        assert len(result.trajectory) > 1

        # Energy should generally decrease (or stay same)
        energies = [t["energy"] for t in result.trajectory]
        for i in range(1, len(energies)):
            # Allow small increases due to noise and numerical precision
            assert energies[i] <= energies[i-1] + 0.2

    def test_convergence(self, network):
        """Network converges or reaches max steps."""
        pattern = torch.randn(256)
        network.store(pattern, "test")

        result = network.retrieve(pattern, max_steps=100)

        # Should converge (steps <= max) with high confidence
        assert result.steps <= 100
        assert result.confidence > 0.9  # High confidence for exact match

    def test_compute_energy(self, network):
        """Energy computation works."""
        pattern = torch.randn(256)
        network.store(pattern, "test")

        energy = network.compute_energy(pattern)

        assert isinstance(energy, float)

    def test_remove_pattern(self, network):
        """Pattern removal works."""
        pattern = torch.randn(256)
        network.store(pattern, "test")

        assert network.pattern_count == 1

        removed = network.remove("test")

        assert removed == True
        assert network.pattern_count == 0

    def test_remove_nonexistent(self, network):
        """Removing nonexistent pattern returns False."""
        removed = network.remove("nonexistent")
        assert removed == False

    def test_clear(self, network):
        """Clear removes all patterns."""
        for i in range(5):
            network.store(torch.randn(256), f"p{i}")

        assert network.pattern_count == 5

        network.clear()

        assert network.pattern_count == 0
        assert network.W.abs().sum() == 0

    def test_analyze(self, network):
        """Analysis returns correct info."""
        for i in range(5):
            network.store(torch.randn(256), f"p{i}")

        analysis = network.analyze()

        assert analysis["pattern_count"] == 5
        assert analysis["capacity_usage"] == 5 / network.capacity
        assert "weight_matrix_norm" in analysis
        assert "average_pattern_overlap" in analysis

    def test_basin_estimate(self, network):
        """Basin estimation works."""
        pattern = torch.randn(256)
        network.store(pattern, "test")

        basin = network.get_basin_estimate(
            "test",
            num_samples=10,
            noise_levels=[0.1, 0.3, 0.5]
        )

        assert len(basin) == 3
        for noise, accuracy in basin.items():
            assert 0.0 <= accuracy <= 1.0

    def test_pattern_orthogonality_affects_retrieval(self):
        """More orthogonal patterns are easier to retrieve."""
        network = AttractorNetwork(dim=256)

        # Store orthogonal patterns
        patterns = []
        for i in range(5):
            p = torch.zeros(256)
            p[i*50:(i+1)*50] = torch.randn(50)  # Non-overlapping
            patterns.append(p)
            network.store(p, f"p{i}")

        # Test retrieval with noisy cues
        correct = 0
        for i, p in enumerate(patterns):
            noisy = p + 0.2 * torch.randn(256)
            result = network.retrieve(noisy)
            if result.pattern_id == f"p{i}":
                correct += 1

        # Should get most of them right
        assert correct >= 3

    def test_usage_ratio(self, network):
        """Usage ratio computed correctly."""
        assert network.usage_ratio == 0.0

        for i in range(10):
            network.store(torch.randn(256), f"p{i}")

        expected = 10 / network.capacity
        assert abs(network.usage_ratio - expected) < 0.001


class TestModernHopfieldNetwork:
    """Tests for modern Hopfield network."""

    @pytest.fixture
    def network(self):
        """Create modern Hopfield network."""
        return ModernHopfieldNetwork(
            dim=256,
            beta=1.0,
            settling_steps=10
        )

    def test_initialization(self, network):
        """Network initializes correctly."""
        assert network.dim == 256
        assert network.beta == 1.0

    def test_store_and_retrieve(self, network):
        """Basic store and retrieve works."""
        pattern = torch.randn(256)
        network.store(pattern, "test")

        result = network.retrieve(pattern)

        assert result.pattern_id == "test"
        assert result.confidence > 0.5

    def test_retrieve_empty(self, network):
        """Retrieval on empty network handles gracefully."""
        cue = torch.randn(256)

        result = network.retrieve(cue)

        assert result.pattern_id is None
        assert result.confidence == 0.0

    def test_softmax_retrieval(self, network):
        """Softmax-based retrieval works."""
        # Store multiple patterns
        for i in range(5):
            network.store(torch.randn(256), f"p{i}")

        cue = torch.randn(256)
        result = network.retrieve(cue)

        assert result.pattern_id is not None
        assert 0 <= result.confidence <= 1
