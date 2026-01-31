"""
Tests for Activity-Dependent Neurogenesis (Phase 4A).

Validates biologically-inspired neuron birth/death mechanisms based on:
- Kempermann (2015): Adult hippocampal neurogenesis
- Tashiro (2007): Experience-dependent neuron survival
- Schmidt-Hieber (2004): Enhanced plasticity in immature neurons
"""

import numpy as np
import pytest

from t4dm.encoding.neurogenesis import (
    NeurogenesisConfig,
    NeurogenesisManager,
    NeurogenesisState,
    NeuronMetadata,
    NeuronState,
)
from t4dm.nca.forward_forward import ForwardForwardConfig, ForwardForwardLayer


class TestNeurogenesisConfig:
    """Test NeurogenesisConfig dataclass."""

    def test_default_config(self):
        """Default config has sensible biological values."""
        config = NeurogenesisConfig()
        assert config.birth_rate == 0.001  # ~0.1% per event
        assert config.survival_threshold == 0.1
        assert config.maturation_epochs == 10
        assert config.enable_birth is True
        assert config.enable_pruning is True

    def test_custom_config(self):
        """Custom config values are respected."""
        config = NeurogenesisConfig(
            birth_rate=0.002,
            maturation_epochs=20,
            immature_lr_boost=3.0,
        )
        assert config.birth_rate == 0.002
        assert config.maturation_epochs == 20
        assert config.immature_lr_boost == 3.0

    def test_validation(self):
        """Config validates parameters."""
        with pytest.raises(AssertionError):
            NeurogenesisConfig(birth_rate=-0.1)

        with pytest.raises(AssertionError):
            NeurogenesisConfig(birth_rate=1.5)

        with pytest.raises(AssertionError):
            NeurogenesisConfig(maturation_epochs=0)

        with pytest.raises(AssertionError):
            NeurogenesisConfig(
                min_neurons_per_layer=100, max_neurons_per_layer=50
            )


class TestNeuronMetadata:
    """Test NeuronMetadata tracking."""

    def test_initialization(self):
        """Metadata initializes correctly."""
        meta = NeuronMetadata(neuron_idx=5, birth_epoch=10)
        assert meta.neuron_idx == 5
        assert meta.birth_epoch == 10
        assert meta.maturity == 0.0
        assert meta.state == NeuronState.IMMATURE
        assert meta.total_activations == 0

    def test_record_activity(self):
        """Recording activity updates statistics."""
        meta = NeuronMetadata(neuron_idx=0, birth_epoch=0)

        # Record some activations
        for i in range(5):
            meta.record_activity(float(i) * 0.1, epoch=i)

        assert meta.total_activations == 5
        assert len(meta.activity_history) == 5
        assert meta.mean_activity == pytest.approx(0.2, abs=0.01)
        assert meta.last_active_epoch == 4

    def test_activity_history_bounded(self):
        """Activity history is bounded (MEM-007 pattern)."""
        meta = NeuronMetadata(neuron_idx=0, birth_epoch=0)
        meta._max_history = 10

        # Record more than max_history activations
        for i in range(20):
            meta.record_activity(1.0, epoch=i)

        assert len(meta.activity_history) <= meta._max_history

    def test_update_maturity(self):
        """Maturity updates correctly."""
        meta = NeuronMetadata(neuron_idx=0, birth_epoch=0)

        # Immature
        meta.update_maturity(0.2)
        assert meta.maturity == 0.2
        assert meta.state == NeuronState.IMMATURE

        # Maturing
        meta.update_maturity(0.3)
        assert meta.maturity == 0.5
        assert meta.state == NeuronState.MATURING

        # Mature
        meta.update_maturity(0.5)
        assert meta.maturity == 1.0
        assert meta.state == NeuronState.MATURE

        # Can't exceed 1.0
        meta.update_maturity(0.5)
        assert meta.maturity == 1.0

    def test_lr_multiplier(self):
        """Learning rate multiplier decreases with maturity."""
        config = NeurogenesisConfig(immature_lr_boost=2.0)
        meta = NeuronMetadata(neuron_idx=0, birth_epoch=0)

        # Immature: high boost
        meta.maturity = 0.0
        lr_mult = meta.get_lr_multiplier(config)
        assert lr_mult == 2.0

        # Maturing: intermediate
        meta.maturity = 0.5
        lr_mult = meta.get_lr_multiplier(config)
        assert lr_mult == 1.5

        # Mature: no boost
        meta.maturity = 1.0
        lr_mult = meta.get_lr_multiplier(config)
        assert lr_mult == 1.0

    def test_to_dict(self):
        """Metadata serializes to dict."""
        meta = NeuronMetadata(neuron_idx=3, birth_epoch=5)
        meta.maturity = 0.7
        meta.total_activations = 100

        d = meta.to_dict()
        assert d["neuron_idx"] == 3
        assert d["birth_epoch"] == 5
        assert d["maturity"] == 0.7
        assert d["total_activations"] == 100


class TestNeurogenesisState:
    """Test NeurogenesisState tracking."""

    def test_initialization(self):
        """State initializes correctly."""
        state = NeurogenesisState()
        assert state.total_births == 0
        assert state.total_deaths == 0
        assert state.current_epoch == 0

    def test_to_dict(self):
        """State serializes to dict."""
        state = NeurogenesisState()
        state.total_births = 10
        state.total_deaths = 5
        state.novelty_history = [0.5, 0.6, 0.7]

        d = state.to_dict()
        assert d["total_births"] == 10
        assert d["total_deaths"] == 5
        assert "mean_novelty" in d

    def test_history_trimming(self):
        """History is trimmed to prevent unbounded growth."""
        state = NeurogenesisState()
        state._max_history = 10

        # Add more than max
        for i in range(20):
            state.novelty_history.append(float(i))
            state.birth_history.append(1)
            state.death_history.append(0)
            state.neuron_count_history.append(100 + i)

        state._trim_history()

        assert len(state.novelty_history) <= state._max_history
        assert len(state.birth_history) <= state._max_history
        assert len(state.death_history) <= state._max_history
        assert len(state.neuron_count_history) <= state._max_history


class TestNeurogenesisManager:
    """Test NeurogenesisManager core functionality."""

    @pytest.fixture
    def config(self):
        """Standard test config."""
        return NeurogenesisConfig(
            birth_rate=0.99,  # High rate for testing
            novelty_threshold=1.0,
            min_novelty_score=0.1,
            survival_threshold=0.1,
            maturation_epochs=10,
            max_neurons_per_layer=100,
            min_neurons_per_layer=10,
            pruning_interval=5,
        )

    @pytest.fixture
    def manager(self, config):
        """Create manager."""
        return NeurogenesisManager(config, random_seed=42)

    @pytest.fixture
    def layer(self):
        """Create a small FF layer for testing."""
        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        return ForwardForwardLayer(layer_config, layer_idx=0, random_seed=42)

    def test_initialization(self, manager):
        """Manager initializes correctly."""
        assert isinstance(manager.config, NeurogenesisConfig)
        assert isinstance(manager.state, NeurogenesisState)
        assert len(manager._neuron_metadata) == 0

    def test_maybe_add_neuron_high_novelty(self, manager, layer):
        """High novelty triggers neuron birth."""
        initial_neurons = layer.config.hidden_dim

        # High novelty should trigger birth (with high birth_rate)
        added = manager.maybe_add_neuron(
            novelty_score=5.0, layer=layer, epoch=0
        )

        assert added is True
        assert layer.config.hidden_dim == initial_neurons + 1
        assert manager.state.total_births == 1

        # Verify neuron metadata tracked
        assert layer.layer_idx in manager._neuron_metadata
        assert initial_neurons in manager._neuron_metadata[layer.layer_idx]

    def test_maybe_add_neuron_low_novelty(self, manager, layer):
        """Low novelty doesn't trigger birth."""
        initial_neurons = layer.config.hidden_dim

        # Low novelty shouldn't trigger birth
        added = manager.maybe_add_neuron(
            novelty_score=0.01, layer=layer, epoch=0
        )

        assert added is False
        assert layer.config.hidden_dim == initial_neurons
        assert manager.state.total_births == 0

    def test_neuron_birth_respects_capacity(self, manager, layer):
        """Birth respects max capacity."""
        # Set layer to near max capacity
        manager.config.max_neurons_per_layer = layer.config.hidden_dim + 1

        # First birth should work
        added1 = manager.maybe_add_neuron(
            novelty_score=10.0, layer=layer, epoch=0
        )
        assert added1 is True

        # Second birth should fail (at capacity)
        added2 = manager.maybe_add_neuron(
            novelty_score=10.0, layer=layer, epoch=0
        )
        assert added2 is False

    def test_neuron_birth_disabled(self, config, layer):
        """Birth can be disabled via config."""
        config.enable_birth = False
        manager = NeurogenesisManager(config, random_seed=42)

        added = manager.maybe_add_neuron(
            novelty_score=10.0, layer=layer, epoch=0
        )

        assert added is False

    def test_update_activity(self, manager, layer):
        """Activity tracking works."""
        # Add a neuron and track it
        manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)

        # Create some activations
        activations = np.random.randn(layer.config.hidden_dim).astype(np.float32)

        # Update activity
        manager.update_activity(layer, activations, epoch=0)

        # Verify tracking
        metadata = manager._neuron_metadata[layer.layer_idx]
        for neuron_idx in metadata:
            assert metadata[neuron_idx].total_activations == 1

    def test_update_maturation(self, manager, layer):
        """Maturation updates over epochs."""
        # Add a neuron
        manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)
        neuron_idx = layer.config.hidden_dim - 1

        # Get initial maturity
        metadata = manager._neuron_metadata[layer.layer_idx][neuron_idx]
        initial_maturity = metadata.maturity

        # Update maturation for several epochs
        for epoch in range(5):
            manager.update_maturation(epoch)

        # Verify maturity increased
        assert metadata.maturity > initial_maturity

    def test_prune_inactive_neurons(self, manager, layer):
        """Inactive neurons are pruned."""
        # Add neurons
        for _ in range(3):
            manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)

        initial_count = layer.config.hidden_dim

        # Record low activity for some neurons
        for epoch in range(20):
            # Low activations
            activations = np.ones(layer.config.hidden_dim) * 0.01
            manager.update_activity(layer, activations, epoch=epoch)

            # Age neurons
            manager.update_maturation(epoch)

        # Prune (force=True to bypass interval)
        pruned = manager.prune_inactive(layer, epoch=20, force=True)

        # Should have pruned some neurons
        assert pruned > 0
        assert layer.config.hidden_dim < initial_count
        assert manager.state.total_deaths == pruned

    def test_pruning_respects_minimum(self, manager, layer):
        """Pruning respects minimum neuron count."""
        # Set layer near minimum
        manager.config.min_neurons_per_layer = layer.config.hidden_dim - 1

        # Record low activity
        for epoch in range(20):
            activations = np.ones(layer.config.hidden_dim) * 0.01
            manager.update_activity(layer, activations, epoch=epoch)
            manager.update_maturation(epoch)

        # Prune
        pruned = manager.prune_inactive(layer, epoch=20, force=True)

        # Should not prune below minimum
        assert layer.config.hidden_dim >= manager.config.min_neurons_per_layer

    def test_pruning_disabled(self, config, layer):
        """Pruning can be disabled via config."""
        config.enable_pruning = False
        manager = NeurogenesisManager(config, random_seed=42)

        # Add neurons
        manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)
        initial_count = layer.config.hidden_dim

        # Record low activity
        for epoch in range(20):
            activations = np.ones(layer.config.hidden_dim) * 0.01
            manager.update_activity(layer, activations, epoch=epoch)

        # Try to prune
        pruned = manager.prune_inactive(layer, epoch=20, force=True)

        assert pruned == 0
        assert layer.config.hidden_dim == initial_count

    def test_pruning_interval(self, manager, layer):
        """Pruning respects interval."""
        manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)

        # Record low activity
        for epoch in range(3):
            activations = np.ones(layer.config.hidden_dim) * 0.01
            manager.update_activity(layer, activations, epoch=epoch)
            manager.update_maturation(epoch)

        # Prune at epoch 3 (before interval)
        pruned = manager.prune_inactive(layer, epoch=3, force=False)

        # Should not prune (interval not reached)
        assert pruned == 0

    def test_get_neuron_lr_multiplier(self, manager, layer):
        """Learning rate multiplier retrieval works."""
        # Add neuron
        manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)
        neuron_idx = layer.config.hidden_dim - 1

        # Immature neuron should have high LR multiplier
        lr_mult = manager.get_neuron_lr_multiplier(layer.layer_idx, neuron_idx)
        assert lr_mult > 1.0

        # Mature it
        metadata = manager._neuron_metadata[layer.layer_idx][neuron_idx]
        metadata.maturity = 1.0

        # Mature neuron should have LR multiplier of 1.0
        lr_mult = manager.get_neuron_lr_multiplier(layer.layer_idx, neuron_idx)
        assert lr_mult == 1.0

    def test_get_neuron_lr_multiplier_missing(self, manager):
        """Missing neuron returns default multiplier."""
        lr_mult = manager.get_neuron_lr_multiplier(layer_idx=999, neuron_idx=999)
        assert lr_mult == 1.0

    def test_get_layer_stats(self, manager, layer):
        """Layer statistics are computed."""
        # Add neurons with different maturities
        for i in range(3):
            manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=i)

        # Set different maturity levels
        metadata = manager._neuron_metadata[layer.layer_idx]
        neurons = list(metadata.keys())
        metadata[neurons[0]].maturity = 0.2
        metadata[neurons[0]].state = NeuronState.IMMATURE
        metadata[neurons[1]].maturity = 0.5
        metadata[neurons[1]].state = NeuronState.MATURING
        metadata[neurons[2]].maturity = 0.9
        metadata[neurons[2]].state = NeuronState.MATURE

        stats = manager.get_layer_stats(layer.layer_idx)

        assert stats["tracked_neurons"] == 3
        assert stats["immature"] == 1
        assert stats["maturing"] == 1
        assert stats["mature"] == 1
        assert "mean_maturity" in stats

    def test_get_stats(self, manager, layer):
        """Global statistics are computed."""
        # Add neurons
        for _ in range(2):
            manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)

        stats = manager.get_stats()

        assert "state" in stats
        assert "total_tracked_neurons" in stats
        assert "layers_tracked" in stats
        assert "config" in stats
        assert stats["total_tracked_neurons"] == 2
        assert stats["layers_tracked"] == 1

    def test_reset(self, manager, layer):
        """Reset clears all state."""
        # Add neurons
        manager.maybe_add_neuron(novelty_score=10.0, layer=layer, epoch=0)
        assert manager.state.total_births == 1

        # Reset
        manager.reset()

        assert manager.state.total_births == 0
        assert len(manager._neuron_metadata) == 0


class TestNeurogenesisIntegration:
    """Test neurogenesis integration with FF layers."""

    def test_neuron_birth_adds_weights(self):
        """Adding neuron properly extends weight matrix."""
        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        initial_hidden = layer.config.hidden_dim
        initial_W_shape = layer.W.shape

        # Add neuron
        new_weights = np.random.randn(32).astype(np.float32)
        new_idx = layer.add_neuron(new_weights)

        assert new_idx == initial_hidden
        assert layer.config.hidden_dim == initial_hidden + 1
        assert layer.W.shape == (32, initial_hidden + 1)
        assert layer.b.shape == (initial_hidden + 1,)

    def test_neuron_removal_reduces_weights(self):
        """Removing neuron properly reduces weight matrix."""
        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        initial_hidden = layer.config.hidden_dim

        # Remove middle neuron
        layer.remove_neuron(neuron_idx=8)

        assert layer.config.hidden_dim == initial_hidden - 1
        assert layer.W.shape == (32, initial_hidden - 1)
        assert layer.b.shape == (initial_hidden - 1,)

    def test_forward_pass_after_growth(self):
        """Forward pass works after adding neurons."""
        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        # Add neurons
        for _ in range(3):
            weights = np.random.randn(32).astype(np.float32)
            layer.add_neuron(weights)

        # Forward pass should work
        x = np.random.randn(32).astype(np.float32)
        output = layer.forward(x)

        assert output.shape == (19,)  # 16 + 3

    def test_forward_pass_after_pruning(self):
        """Forward pass works after removing neurons."""
        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        # Remove neurons
        layer.remove_neuron(10)
        layer.remove_neuron(5)

        # Forward pass should work
        x = np.random.randn(32).astype(np.float32)
        output = layer.forward(x)

        assert output.shape == (14,)  # 16 - 2

    def test_learning_after_neurogenesis(self):
        """Learning works after neuron birth/death."""
        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        # Add neuron
        weights = np.random.randn(32).astype(np.float32)
        layer.add_neuron(weights)

        # Learn
        x = np.random.randn(32).astype(np.float32)
        h = layer.forward(x, training=True)
        stats = layer.learn_positive(x, h)

        assert stats["phase"] == "positive"
        assert stats["goodness"] > 0

    def test_neuron_count_stabilizes(self):
        """Neuron count stabilizes over time with birth and death."""
        config = NeurogenesisConfig(
            birth_rate=0.01,
            survival_threshold=0.2,
            maturation_epochs=5,
            max_neurons_per_layer=50,
            min_neurons_per_layer=10,
        )
        manager = NeurogenesisManager(config, random_seed=42)

        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=20)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        counts = []

        # Simulate training
        for epoch in range(30):
            # Randomly try to add neurons
            novelty = np.random.uniform(0, 2.0)
            manager.maybe_add_neuron(novelty, layer, epoch)

            # Simulate activity
            activations = np.random.uniform(0, 1.0, layer.config.hidden_dim)
            manager.update_activity(layer, activations, epoch)

            # Mature neurons
            manager.update_maturation(epoch)

            # Prune periodically
            if epoch % 5 == 0:
                manager.prune_inactive(layer, epoch, force=True)

            counts.append(layer.config.hidden_dim)

        # Verify count remains bounded
        assert min(counts) >= config.min_neurons_per_layer
        assert max(counts) <= config.max_neurons_per_layer


class TestNeurogenesisBiologicalValidity:
    """Test biological plausibility of neurogenesis implementation."""

    def test_immature_neurons_higher_plasticity(self):
        """Immature neurons have enhanced learning rates (Schmidt-Hieber 2004)."""
        config = NeurogenesisConfig(immature_lr_boost=2.0)
        meta_immature = NeuronMetadata(neuron_idx=0, birth_epoch=0)
        meta_immature.maturity = 0.0

        meta_mature = NeuronMetadata(neuron_idx=1, birth_epoch=0)
        meta_mature.maturity = 1.0

        lr_immature = meta_immature.get_lr_multiplier(config)
        lr_mature = meta_mature.get_lr_multiplier(config)

        assert lr_immature > lr_mature

    def test_novelty_enhances_neurogenesis(self):
        """Novel experiences increase birth rate (Gould 1999)."""
        config = NeurogenesisConfig(birth_rate=0.5)
        manager = NeurogenesisManager(config, random_seed=42)

        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        # Record normal novelty
        for _ in range(10):
            manager.maybe_add_neuron(0.5, layer, epoch=0)

        births_normal = manager.state.total_births

        # Reset
        manager.reset()
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        # Record high novelty
        for _ in range(10):
            manager.maybe_add_neuron(5.0, layer, epoch=0)

        births_high = manager.state.total_births

        # High novelty should produce more births
        assert births_high >= births_normal

    def test_inactive_neurons_pruned(self):
        """Inactive neurons undergo apoptosis (Tashiro 2007)."""
        config = NeurogenesisConfig(survival_threshold=0.5, min_neurons_per_layer=5)
        manager = NeurogenesisManager(config, random_seed=42)
        config.birth_rate = 0.99  # High rate to ensure neurons are actually added

        layer_config = ForwardForwardConfig(input_dim=32, hidden_dim=16)
        layer = ForwardForwardLayer(layer_config, random_seed=42)

        # Add neurons
        for _ in range(5):
            manager.maybe_add_neuron(10.0, layer, epoch=0)

        # Record low activity
        for epoch in range(20):
            # Very low activations
            activations = np.ones(layer.config.hidden_dim) * 0.01
            manager.update_activity(layer, activations, epoch)
            manager.update_maturation(epoch)

        initial_count = layer.config.hidden_dim

        # Prune
        manager.prune_inactive(layer, epoch=20, force=True)

        # Should have pruned inactive neurons
        assert layer.config.hidden_dim < initial_count

    def test_maturation_is_gradual(self):
        """Neuron maturation occurs over multiple epochs (biological realism)."""
        config = NeurogenesisConfig(maturation_epochs=10, integration_rate=0.1)

        meta = NeuronMetadata(neuron_idx=0, birth_epoch=0)

        # Mature over multiple epochs
        for _ in range(5):
            meta.update_maturity(config.integration_rate)

        # Should be partially mature, not fully
        assert 0.3 < meta.maturity < 0.7
        assert meta.state == NeuronState.MATURING
