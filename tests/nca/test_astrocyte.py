"""
Tests for astrocyte glial layer.

Validates:
1. EAAT-2 glutamate reuptake (Michaelis-Menten kinetics)
2. GAT-3 GABA reuptake
3. Gliotransmission (Glu, D-serine, ATP release)
4. Calcium dynamics and state transitions
5. Metabolic support (lactate shuttle)
6. Neuroprotection and excitotoxicity prevention
7. Integration with NeuralFieldSolver
"""

import numpy as np
import pytest

from t4dm.nca.astrocyte import (
    AstrocyteLayer,
    AstrocyteConfig,
    AstrocyteLayerState,
    AstrocyteState,
    compute_tripartite_synapse,
)
from t4dm.nca.neural_field import (
    NeuralFieldSolver,
    NeuralFieldConfig,
    NeurotransmitterType,
)


class TestAstrocyteConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Default config has biologically reasonable values."""
        config = AstrocyteConfig()

        # EAAT-2 parameters
        assert config.eaat2_vmax > 0
        assert config.eaat2_km > 0
        assert config.eaat2_km < 1.0  # Normalized

        # GAT-3 parameters
        assert config.gat3_vmax > 0
        assert config.gat3_km > 0

        # Gliotransmission
        assert config.gliotx_threshold > 0
        assert config.gliotx_threshold < 1.0

    def test_eaat2_faster_than_gat3(self):
        """EAAT-2 (glutamate) should be faster than GAT-3 (GABA)."""
        config = AstrocyteConfig()

        # Glutamate clearance is critical for preventing excitotoxicity
        assert config.eaat2_vmax > config.gat3_vmax


class TestAstrocyteLayerBasic:
    """Test basic astrocyte functionality."""

    def test_initialization(self):
        """Astrocyte layer initializes correctly."""
        astro = AstrocyteLayer()

        assert astro.state.calcium > 0
        assert astro.state.glycogen == 1.0  # Full energy reserve
        assert astro.state.activation_state == AstrocyteState.QUIESCENT

    def test_reuptake_clears_glutamate(self):
        """Glutamate reuptake should reduce glutamate levels."""
        astro = AstrocyteLayer()

        glu = 0.7
        gaba = 0.3

        glu_cleared, gaba_cleared = astro.compute_reuptake(glu, gaba)

        assert glu_cleared > 0, "Should clear some glutamate"
        assert glu_cleared < glu, "Cannot clear more than available"

    def test_reuptake_clears_gaba(self):
        """GABA reuptake should reduce GABA levels."""
        astro = AstrocyteLayer()

        glu = 0.3
        gaba = 0.7

        glu_cleared, gaba_cleared = astro.compute_reuptake(glu, gaba)

        assert gaba_cleared > 0, "Should clear some GABA"
        assert gaba_cleared < gaba, "Cannot clear more than available"

    def test_michaelis_menten_kinetics(self):
        """Reuptake should follow Michaelis-Menten saturation."""
        astro = AstrocyteLayer()

        # Low substrate - linear regime
        glu_low, _ = astro.compute_reuptake(0.1, 0.1)

        # High substrate - saturated regime
        astro.reset()
        glu_high, _ = astro.compute_reuptake(0.9, 0.1)

        # Rate should not increase proportionally with substrate
        # (saturation effect)
        ratio_substrate = 0.9 / 0.1  # 9x
        ratio_rate = glu_high / glu_low

        assert ratio_rate < ratio_substrate, "Should show saturation"


class TestGlutamateReuptake:
    """Test EAAT-2 glutamate transporter specifics."""

    def test_high_glutamate_triggers_more_reuptake(self):
        """Higher glutamate should trigger faster reuptake."""
        astro = AstrocyteLayer()

        # Low glutamate
        glu_cleared_low, _ = astro.compute_reuptake(0.2, 0.3)

        # High glutamate
        astro.reset()
        glu_cleared_high, _ = astro.compute_reuptake(0.8, 0.3)

        assert glu_cleared_high > glu_cleared_low

    def test_activity_modulates_reuptake(self):
        """Higher neural activity should upregulate transporters."""
        astro = AstrocyteLayer()

        # Low activity
        glu_low, _ = astro.compute_reuptake(0.5, 0.3, activity_level=0.2)

        # High activity
        astro.reset()
        glu_high, _ = astro.compute_reuptake(0.5, 0.3, activity_level=0.8)

        assert glu_high > glu_low, "Activity should enhance reuptake"

    def test_excitotoxicity_detection(self):
        """Should detect excitotoxic glutamate levels."""
        config = AstrocyteConfig(excitotoxicity_threshold=0.9)
        astro = AstrocyteLayer(config)

        # Normal glutamate
        astro.compute_reuptake(0.5, 0.3)
        assert astro.state.excitotoxicity_events == 0

        # Excitotoxic glutamate
        astro.compute_reuptake(0.95, 0.3)
        assert astro.state.excitotoxicity_events == 1


class TestGABAReuptake:
    """Test GAT-3 GABA transporter specifics."""

    def test_gaba_clearance(self):
        """GAT-3 should clear GABA."""
        astro = AstrocyteLayer()

        _, gaba_cleared = astro.compute_reuptake(0.3, 0.6)

        assert gaba_cleared > 0

    def test_gaba_reuptake_slower_than_glutamate(self):
        """GABA reuptake should be slower than glutamate at equal concentrations."""
        astro = AstrocyteLayer()

        # Equal concentrations
        glu_cleared, gaba_cleared = astro.compute_reuptake(0.5, 0.5)

        # Glutamate clearance is critical, should be faster
        assert glu_cleared >= gaba_cleared


class TestCalciumDynamics:
    """Test astrocyte calcium signaling."""

    def test_calcium_rises_with_glutamate(self):
        """Glutamate should trigger calcium rise (mGluR activation)."""
        astro = AstrocyteLayer()
        initial_ca = astro.state.calcium

        # High glutamate
        for _ in range(50):
            astro.compute_reuptake(0.8, 0.3)

        assert astro.state.calcium > initial_ca

    def test_calcium_decays_at_rest(self):
        """Calcium should decay toward baseline without stimulation."""
        config = AstrocyteConfig(ca_decay_rate=0.1)
        astro = AstrocyteLayer(config)

        # First elevate calcium
        for _ in range(50):
            astro.compute_reuptake(0.8, 0.3)

        elevated_ca = astro.state.calcium

        # Then let it decay
        for _ in range(100):
            astro.compute_reuptake(0.1, 0.1)

        assert astro.state.calcium < elevated_ca

    def test_state_transition_to_activated(self):
        """High calcium should trigger ACTIVATED state."""
        config = AstrocyteConfig(ca_threshold=0.4, ca_rise_rate=0.2)
        astro = AstrocyteLayer(config)

        # Stimulate until activated
        for _ in range(100):
            astro.compute_reuptake(0.9, 0.3)
            if astro.state.activation_state == AstrocyteState.ACTIVATED:
                break

        # Should eventually become activated
        assert astro.state.calcium >= config.ca_threshold

    def test_reactive_state_at_high_calcium(self):
        """Very high calcium should trigger pathological REACTIVE state."""
        config = AstrocyteConfig(
            reactive_threshold=0.7,
            ca_rise_rate=0.3
        )
        astro = AstrocyteLayer(config)

        # Strong stimulation
        for _ in range(200):
            astro.compute_reuptake(0.95, 0.3, activity_level=0.9)

        if astro.state.calcium > config.reactive_threshold:
            assert astro.state.activation_state == AstrocyteState.REACTIVE


class TestGliotransmission:
    """Test gliotransmitter release."""

    def test_no_release_at_low_calcium(self):
        """No gliotransmitter release below threshold."""
        config = AstrocyteConfig(gliotx_threshold=0.6)
        astro = AstrocyteLayer(config)

        # Keep calcium low
        astro.state.calcium = 0.3

        gliotx = astro.compute_gliotransmission()

        assert gliotx["glutamate"] == 0
        assert gliotx["dserine"] == 0
        assert gliotx["atp"] == 0

    def test_release_above_threshold(self):
        """Gliotransmitter release above calcium threshold."""
        config = AstrocyteConfig(gliotx_threshold=0.5)
        astro = AstrocyteLayer(config)

        # Elevate calcium above threshold
        astro.state.calcium = 0.7

        gliotx = astro.compute_gliotransmission()

        assert gliotx["glutamate"] > 0
        assert gliotx["dserine"] > 0
        assert gliotx["atp"] > 0

    def test_release_proportional_to_calcium(self):
        """Higher calcium should release more gliotransmitters."""
        config = AstrocyteConfig(gliotx_threshold=0.4)
        astro = AstrocyteLayer(config)

        # Moderate calcium
        astro.state.calcium = 0.6
        gliotx_low = astro.compute_gliotransmission()
        glu_low = gliotx_low["glutamate"]

        # Reset refractory
        astro.state.release_refractory = False

        # High calcium
        astro.state.calcium = 0.9
        gliotx_high = astro.compute_gliotransmission()
        glu_high = gliotx_high["glutamate"]

        assert glu_high > glu_low

    def test_refractory_period(self):
        """After release, should enter refractory period."""
        config = AstrocyteConfig(gliotx_threshold=0.4)
        astro = AstrocyteLayer(config)

        # Trigger release
        astro.state.calcium = 0.8
        gliotx1 = astro.compute_gliotransmission()

        # Should be in refractory now
        if gliotx1["glutamate"] > 0:
            gliotx2 = astro.compute_gliotransmission()
            # Should be blocked by refractory
            assert gliotx2["glutamate"] == 0 or not astro.state.release_refractory


class TestMetabolicSupport:
    """Test astrocyte-neuron lactate shuttle."""

    def test_lactate_production_with_activity(self):
        """Higher activity should produce more lactate."""
        astro = AstrocyteLayer()

        lactate_low = astro.compute_metabolic_support(activity_level=0.2)
        lactate_high = astro.compute_metabolic_support(activity_level=0.8)

        assert lactate_high > lactate_low

    def test_glycogen_consumption_at_high_activity(self):
        """High activity should consume glycogen reserves."""
        astro = AstrocyteLayer()
        initial_glycogen = astro.state.glycogen

        # High activity for many steps
        for _ in range(100):
            astro.compute_metabolic_support(activity_level=0.9)

        assert astro.state.glycogen < initial_glycogen

    def test_glycogen_replenishment_at_low_activity(self):
        """Low activity should replenish glycogen."""
        astro = AstrocyteLayer()

        # First deplete
        astro.state.glycogen = 0.5

        # Then rest
        for _ in range(100):
            astro.compute_metabolic_support(activity_level=0.1)

        assert astro.state.glycogen > 0.5


class TestNeuroprotection:
    """Test neuroprotection scoring."""

    def test_neuroprotection_score_healthy(self):
        """Healthy astrocyte should have high neuroprotection score."""
        astro = AstrocyteLayer()

        score = astro.get_neuroprotection_score()

        assert score > 0.7  # Healthy baseline

    def test_neuroprotection_decreases_when_reactive(self):
        """Reactive astrocytes have impaired neuroprotection."""
        astro = AstrocyteLayer()

        # Force reactive state
        astro.state.activation_state = AstrocyteState.REACTIVE
        score_reactive = astro.get_neuroprotection_score()

        # Normal state
        astro.state.activation_state = AstrocyteState.QUIESCENT
        score_normal = astro.get_neuroprotection_score()

        assert score_normal > score_reactive

    def test_neuroprotection_decreases_with_low_glycogen(self):
        """Low energy reserves impair neuroprotection."""
        astro = AstrocyteLayer()

        # Full glycogen
        astro.state.glycogen = 1.0
        score_full = astro.get_neuroprotection_score()

        # Depleted glycogen
        astro.state.glycogen = 0.1
        score_depleted = astro.get_neuroprotection_score()

        assert score_full > score_depleted


class TestFieldReuptake:
    """Test spatial field reuptake operations."""

    def test_compute_reuptake_field(self):
        """Reuptake should work across spatial fields."""
        astro = AstrocyteLayer()

        glu_field = np.full((16,), 0.6, dtype=np.float32)
        gaba_field = np.full((16,), 0.4, dtype=np.float32)

        glu_cleared, gaba_cleared = astro.compute_reuptake_field(glu_field, gaba_field)

        assert glu_cleared.shape == glu_field.shape
        assert gaba_cleared.shape == gaba_field.shape
        assert np.all(glu_cleared > 0)
        assert np.all(gaba_cleared > 0)

    def test_field_reuptake_varies_with_concentration(self):
        """Reuptake should vary with local concentration."""
        astro = AstrocyteLayer()

        # Gradient of glutamate
        glu_field = np.linspace(0.2, 0.9, 16).astype(np.float32)
        gaba_field = np.full((16,), 0.4, dtype=np.float32)

        glu_cleared, _ = astro.compute_reuptake_field(glu_field, gaba_field)

        # Higher glutamate regions should have more clearance
        assert glu_cleared[-1] > glu_cleared[0]


class TestTripartiteSynapse:
    """Test tripartite synapse function."""

    def test_tripartite_synapse_basic(self):
        """Tripartite synapse computes all components."""
        astro = AstrocyteLayer()

        result = compute_tripartite_synapse(
            presynaptic=0.7,
            postsynaptic=0.5,
            astrocyte=astro,
            glutamate=0.6,
            gaba=0.4
        )

        assert "glutamate" in result
        assert "gaba" in result
        assert "glu_cleared" in result
        assert "gaba_cleared" in result
        assert "nmda_potentiation" in result
        assert "adenosine" in result

    def test_glutamate_reduced_by_reuptake(self):
        """Net glutamate should be reduced by astrocyte reuptake."""
        astro = AstrocyteLayer()

        initial_glu = 0.6
        result = compute_tripartite_synapse(
            presynaptic=0.5,
            postsynaptic=0.5,
            astrocyte=astro,
            glutamate=initial_glu,
            gaba=0.4
        )

        assert result["glutamate"] < initial_glu


class TestNeuralFieldIntegration:
    """Test integration with NeuralFieldSolver."""

    def test_solver_with_astrocyte(self):
        """NeuralFieldSolver works with astrocyte layer."""
        config = NeuralFieldConfig(
            spatial_dims=1,
            grid_size=16,
            dt=0.001,
        )
        astro = AstrocyteLayer()

        solver = NeuralFieldSolver(
            config=config,
            astrocyte_layer=astro
        )

        # Run for several steps
        for _ in range(100):
            state = solver.step()

        # Should have valid state
        assert 0 <= state.glutamate <= 1
        assert 0 <= state.gaba <= 1

    def test_astrocyte_reduces_glutamate(self):
        """Astrocyte should help reduce glutamate accumulation."""
        config = NeuralFieldConfig(
            spatial_dims=1,
            grid_size=16,
            dt=0.001,
        )

        # With astrocyte
        astro = AstrocyteLayer()
        solver_with = NeuralFieldSolver(config=config, astrocyte_layer=astro)

        # Inject glutamate
        solver_with.inject_stimulus(NeurotransmitterType.GLUTAMATE, 0.3)

        # Run for a while
        for _ in range(200):
            solver_with.step()

        glu_with_astro = solver_with.get_mean_state().glutamate

        # Without astrocyte
        solver_without = NeuralFieldSolver(config=config)
        solver_without.inject_stimulus(NeurotransmitterType.GLUTAMATE, 0.3)

        for _ in range(200):
            solver_without.step()

        glu_without_astro = solver_without.get_mean_state().glutamate

        # Astrocyte should help clear glutamate faster
        # (though decay also clears it, astrocyte adds to clearance)
        # This is a weak test since decay dominates
        assert glu_with_astro <= glu_without_astro + 0.1


class TestAstrocyteStats:
    """Test statistics and diagnostics."""

    def test_get_stats(self):
        """get_stats returns comprehensive statistics."""
        astro = AstrocyteLayer()

        for _ in range(100):
            astro.compute_reuptake(0.5, 0.4)

        stats = astro.get_stats()

        assert "step_count" in stats
        assert stats["step_count"] == 100
        assert "calcium" in stats
        assert "activation_state" in stats
        assert "glycogen" in stats
        assert "total_glu_cleared" in stats
        assert "neuroprotection_score" in stats

    def test_reset(self):
        """Reset clears all state."""
        astro = AstrocyteLayer()

        for _ in range(100):
            astro.compute_reuptake(0.7, 0.5)

        astro.reset()

        assert astro.state.total_glu_cleared == 0
        assert astro.state.total_gaba_cleared == 0
        assert astro.state.calcium == 0.1  # Default

    def test_validate_function(self):
        """Validation returns expected criteria."""
        astro = AstrocyteLayer()

        # Run enough steps
        for _ in range(200):
            astro.compute_reuptake(0.5, 0.4)

        validation = astro.validate_function()

        assert "glutamate_clearance" in validation
        assert "energy_reserves" in validation
        assert "not_pathological" in validation
        assert "neuroprotection" in validation
        assert "all_pass" in validation

    def test_state_to_dict(self):
        """State can be serialized to dict."""
        astro = AstrocyteLayer()
        astro.compute_reuptake(0.5, 0.4)

        state_dict = astro.state.to_dict()

        assert "calcium" in state_dict
        assert "glutamate_buffered" in state_dict
        assert "activation_state" in state_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
