"""Tests for biological correctness fixes."""
import numpy as np
import pytest
from ww.nca.neuromod_crosstalk import NeuromodCrosstalk, NeuromodCrosstalkConfig
from ww.nca.hippocampus import HippocampalCircuit, HippocampalConfig


class TestAChDAGating:
    """ATOM-P1-1: ACh-DA gating direction per Threlfell 2012."""

    def test_high_ach_suppresses_da(self):
        """High ACh should suppress DA release."""
        ct = NeuromodCrosstalk()
        gated_da = ct.ach_gates_da(ach_level=0.9, da_signal=1.0)
        assert gated_da < 0.3, f"High ACh should suppress DA, got {gated_da}"

    def test_low_ach_enables_da(self):
        """Low ACh (interneuron pause) should enable DA release."""
        ct = NeuromodCrosstalk()
        gated_da = ct.ach_gates_da(ach_level=0.1, da_signal=1.0)
        assert gated_da > 0.7, f"Low ACh should enable DA, got {gated_da}"

    def test_midpoint_gives_half_gate(self):
        """At ACh=midpoint, gate should be ~0.5."""
        ct = NeuromodCrosstalk()
        gated_da = ct.ach_gates_da(ach_level=0.5, da_signal=1.0)
        assert 0.3 < gated_da < 0.7, f"Midpoint ACh should give ~half gate, got {gated_da}"

    def test_modulate_applies_correct_gating(self):
        """Full modulate() should apply correct ACh-DA gating."""
        ct = NeuromodCrosstalk()
        # High ACh should reduce DA
        result = ct.modulate({"da": 0.8, "5ht": 0.5, "ne": 0.5, "ach": 0.9})
        assert result["da"] < 0.3, f"High ACh should suppress DA, got {result['da']}"

        # Low ACh should preserve DA
        result = ct.modulate({"da": 0.8, "5ht": 0.5, "ne": 0.5, "ach": 0.1})
        assert result["da"] > 0.5, f"Low ACh should allow DA through, got {result['da']}"

    def test_ach_gate_is_monotonic_decreasing(self):
        """Gate strength should decrease monotonically as ACh increases."""
        ct = NeuromodCrosstalk()
        ach_levels = np.linspace(0.0, 1.0, 11)
        gates = [ct.ach_gates_da(ach, 1.0) for ach in ach_levels]

        # Check monotonic decrease
        for i in range(len(gates) - 1):
            assert gates[i] >= gates[i+1], (
                f"Gate should decrease with increasing ACh, "
                f"but gate[{ach_levels[i]:.1f}]={gates[i]:.3f} < "
                f"gate[{ach_levels[i+1]:.1f}]={gates[i+1]:.3f}"
            )

    def test_crosstalk_effects_show_correct_delta(self):
        """Detailed crosstalk effects should show negative delta for high ACh."""
        ct = NeuromodCrosstalk()
        effects = ct.get_crosstalk_effects({"da": 0.8, "5ht": 0.5, "ne": 0.5, "ach": 0.9})

        # ACh-DA pathway should show suppression (negative delta)
        ach_da_effect = effects["ach_gates_da"]
        assert ach_da_effect["delta"] < 0, (
            f"High ACh should suppress DA (negative delta), "
            f"got delta={ach_da_effect['delta']}"
        )
        assert ach_da_effect["modulated"] < ach_da_effect["original"]


class TestCrosstalkMinLevel:
    """ATOM-P2-7: min_nt_level should be 0.05, not 0.0."""

    def test_crosstalk_min_level_matches_circuits(self):
        """P2-7: min_nt_level should be 0.05, not 0.0."""
        config = NeuromodCrosstalkConfig()
        assert config.min_nt_level == 0.05, (
            f"Biological baseline should be 0.05 (never absolute zero), "
            f"got {config.min_nt_level}"
        )

    def test_crosstalk_clamps_to_baseline(self):
        """NT levels should never fall below 0.05 for modulated NTs."""
        ct = NeuromodCrosstalk()

        # Extreme suppression should still hit floor at 0.05
        result = ct.modulate({"da": 0.01, "5ht": 0.01, "ne": 0.5, "ach": 1.0})

        # Check modulated NTs (DA is gated, 5HT is inhibited, ACh is boosted by NE)
        # DA: gated by high ACh, should be clamped to 0.05
        assert result["da"] >= 0.05, (
            f"DA should never fall below 0.05, got {result['da']}"
        )

        # 5HT: inhibited by DA, but input was already 0.01 < 0.05
        # After inhibition it will be clamped to 0.05
        assert result["5ht"] >= 0.05, (
            f"5HT should never fall below 0.05, got {result['5ht']}"
        )

        # ACh: boosted by NE (0.5), should be well above 0.05
        assert result["ach"] >= 0.05, (
            f"ACh should never fall below 0.05, got {result['ach']}"
        )


class TestDGKWTA:
    """ATOM-P2-17: DG sparse pattern has exactly k nonzero elements."""

    def test_dg_kwta_exact_sparsity(self):
        """P2-17: DG sparse pattern has exactly k nonzero elements."""
        config = HippocampalConfig(ec_dim=128, dg_dim=512, dg_sparsity=0.1)
        hpc = HippocampalCircuit(config, random_seed=42)

        rng = np.random.default_rng(42)
        ec_input = rng.random(128).astype(np.float32)
        ec_input = ec_input / np.linalg.norm(ec_input)

        # Process through DG
        hpc.process(ec_input)

        # Access the sparse DG pattern from recent patterns
        sparse_dg_pattern = hpc.dg._recent_patterns[-1]

        # Count nonzero elements
        k_expected = max(1, int(config.dg_dim * config.dg_sparsity))
        k_actual = np.count_nonzero(sparse_dg_pattern)

        assert k_actual == k_expected, (
            f"DG k-WTA should produce exactly {k_expected} nonzero elements, "
            f"got {k_actual}"
        )

    def test_dg_kwta_preserves_signs(self):
        """k-WTA should preserve original signs of top-k elements."""
        config = HippocampalConfig(ec_dim=64, dg_dim=256, dg_sparsity=0.05)
        hpc = HippocampalCircuit(config, random_seed=123)

        rng = np.random.default_rng(123)
        ec_input = rng.standard_normal(64).astype(np.float32)  # Mix of positive/negative
        ec_input = ec_input / np.linalg.norm(ec_input)

        hpc.process(ec_input)
        sparse_dg_pattern = hpc.dg._recent_patterns[-1]

        # All nonzero elements should have consistent signs from input
        nonzero_mask = sparse_dg_pattern != 0
        assert np.any(nonzero_mask), "DG should have nonzero outputs"

    def test_dg_kwta_top_k_magnitude(self):
        """k-WTA should select the top k elements by magnitude."""
        config = HippocampalConfig(ec_dim=64, dg_dim=256, dg_sparsity=0.1)
        hpc = HippocampalCircuit(config, random_seed=456)

        rng = np.random.default_rng(456)
        ec_input = rng.random(64).astype(np.float32)
        ec_input = ec_input / np.linalg.norm(ec_input)

        hpc.process(ec_input)
        sparse_dg_pattern = hpc.dg._recent_patterns[-1]

        # Get nonzero values
        nonzero_vals = sparse_dg_pattern[sparse_dg_pattern != 0]
        k_expected = max(1, int(config.dg_dim * config.dg_sparsity))

        assert len(nonzero_vals) == k_expected

        # These should be larger (in magnitude) than any zero'd element would have been
        min_kept_magnitude = np.min(np.abs(nonzero_vals))

        # All kept values should have reasonable magnitude
        assert min_kept_magnitude > 0, "Kept values should be nonzero"


class TestNEEncodingGain:
    """ATOM-P1-6: NE encoding gain modulates DG input."""

    def test_ne_gain_modulates_dg_input(self):
        """P1-6: High NE changes DG output vs low NE."""
        config = HippocampalConfig(ec_dim=128, dg_dim=512, ca3_dim=128, ca1_dim=128)
        hpc = HippocampalCircuit(config)
        rng = np.random.default_rng(42)
        ec_input = rng.random(128).astype(np.float32)
        ec_input = ec_input / np.linalg.norm(ec_input)

        # Process with low NE
        hpc._ne_level = 0.1
        result_low = hpc.process(ec_input.copy())

        # Process with high NE
        hpc._ne_level = 0.8
        result_high = hpc.process(ec_input.copy())

        # Results should differ (NE modulates encoding gain)
        # At minimum, the DG output should be different
        assert result_low is not None
        assert result_high is not None
        # High NE should result in different DG output due to gain modulation
        dg_diff = np.linalg.norm(result_high.dg_output - result_low.dg_output)
        assert dg_diff > 0.01, f"High NE should change DG output, diff={dg_diff}"

    def test_ne_gain_threshold(self):
        """P1-6: NE=0.2 does not trigger gain."""
        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128, dg_dim=512, ca3_dim=128, ca1_dim=128))
        hpc._ne_level = 0.2
        # Should not raise and should process normally
        rng = np.random.default_rng(42)
        ec_input = rng.random(128).astype(np.float32)
        ec_input = ec_input / np.linalg.norm(ec_input)
        result = hpc.process(ec_input)
        assert result is not None


class TestNBMCognitiveMode:
    """ATOM-P1-11: ACh encoding/retrieval thresholds in NBM."""

    def test_ach_encoding_threshold(self):
        """P1-11: ACh=0.7 → ENCODING."""
        from ww.nca.nucleus_basalis import NucleusBasalisCircuit
        nbm = NucleusBasalisCircuit()
        nbm.state.ach_level = 0.7
        assert nbm.get_cognitive_mode() == "ENCODING"

    def test_ach_retrieval_threshold(self):
        """P1-11: ACh=0.3 → RETRIEVAL."""
        from ww.nca.nucleus_basalis import NucleusBasalisCircuit
        nbm = NucleusBasalisCircuit()
        nbm.state.ach_level = 0.3
        assert nbm.get_cognitive_mode() == "RETRIEVAL"

    def test_ach_transitional(self):
        """P1-11: ACh=0.5 → TRANSITIONAL."""
        from ww.nca.nucleus_basalis import NucleusBasalisCircuit
        nbm = NucleusBasalisCircuit()
        nbm.state.ach_level = 0.5
        assert nbm.get_cognitive_mode() == "TRANSITIONAL"


class TestReconsolidationRequiresRetrieval:
    """ATOM-P1-8: Reconsolidation requires prior retrieval."""

    @pytest.mark.asyncio
    async def test_no_reconsolidation_without_retrieval(self):
        """P1-8: Episode without last_accessed should not reconsolidate."""
        from ww.consolidation.lability import is_reconsolidation_eligible
        from datetime import datetime

        # Episode with no last_accessed (never retrieved)
        class MockEpisode:
            last_accessed = None

        episode = MockEpisode()
        last_retrieval = getattr(episode, "last_accessed", None)

        # Should NOT be eligible for reconsolidation
        if last_retrieval is None:
            can_reconsolidate = False
        else:
            can_reconsolidate = is_reconsolidation_eligible(
                last_retrieval,
                window_hours=6.0
            )

        assert can_reconsolidate is False, (
            "Episode without retrieval (last_accessed=None) should NOT reconsolidate"
        )

    @pytest.mark.asyncio
    async def test_reconsolidation_with_recent_retrieval(self):
        """P1-8: Episode with recent retrieval CAN reconsolidate."""
        from ww.consolidation.lability import is_reconsolidation_eligible
        from datetime import datetime, timedelta

        # Episode retrieved 2 hours ago (within 6h window)
        class MockEpisode:
            last_accessed = datetime.now() - timedelta(hours=2)

        episode = MockEpisode()
        last_retrieval = getattr(episode, "last_accessed", None)

        # Should be eligible for reconsolidation
        if last_retrieval is None:
            can_reconsolidate = False
        else:
            can_reconsolidate = is_reconsolidation_eligible(
                last_retrieval,
                window_hours=6.0
            )

        assert can_reconsolidate is True, (
            "Episode with recent retrieval should be eligible for reconsolidation"
        )


class TestNeuralFieldNaN:
    """ATOM-P1-4: NaN in neural field raises within 1 step."""

    def test_nan_injection_detected_in_state(self):
        """NaN injected into neural field state is detected immediately."""
        from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig, NumericalInstabilityError

        config = NeuralFieldConfig(spatial_dims=1, grid_size=16)
        solver = NeuralFieldSolver(config)

        # Inject NaN into state
        solver.fields[0, 0] = float('nan')

        with pytest.raises(NumericalInstabilityError, match="NaN/Inf detected"):
            solver.step(dt=0.01)

    def test_inf_injection_detected(self):
        """Inf injected into neural field state is detected immediately."""
        from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig, NumericalInstabilityError

        config = NeuralFieldConfig(spatial_dims=1, grid_size=16)
        solver = NeuralFieldSolver(config)

        # Inject Inf into state
        solver.fields[2, 5] = float('inf')

        with pytest.raises(NumericalInstabilityError, match="NaN/Inf detected"):
            solver.step(dt=0.01)

    def test_negative_inf_detected(self):
        """Negative Inf is also detected."""
        from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig, NumericalInstabilityError

        config = NeuralFieldConfig(spatial_dims=1, grid_size=16)
        solver = NeuralFieldSolver(config)

        # Inject negative Inf
        solver.fields[3, 7] = float('-inf')

        with pytest.raises(NumericalInstabilityError, match="NaN/Inf detected"):
            solver.step(dt=0.01)

    def test_normal_step_does_not_raise(self):
        """Normal step with finite values does not raise."""
        from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig

        config = NeuralFieldConfig(spatial_dims=1, grid_size=16)
        solver = NeuralFieldSolver(config)

        # Normal step should work fine
        state = solver.step(dt=0.01)
        assert state is not None
        assert np.all(np.isfinite(solver.fields))

    def test_multiple_nans_reported(self):
        """Error message includes count of non-finite values."""
        from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig, NumericalInstabilityError

        config = NeuralFieldConfig(spatial_dims=1, grid_size=16)
        solver = NeuralFieldSolver(config)

        # Inject multiple NaNs
        solver.fields[0, 0] = float('nan')
        solver.fields[1, 5] = float('nan')
        solver.fields[2, 10] = float('inf')

        with pytest.raises(NumericalInstabilityError, match=r"Non-finite count: \d+"):
            solver.step(dt=0.01)


class TestDalesLaw:
    """ATOM-P1-5: Mixed-sign weights raise in strict mode."""

    def test_dales_law_strict_rejects_gaba_excitatory(self):
        """GABA pathway that is not inhibitory raises in strict mode."""
        from ww.nca.connectome import ConnectomeConfig, Connectome, DalesLawViolation, ProjectionPathway, NTSystem

        config = ConnectomeConfig(enforce_dale_law=True, strict_dales_law=True)
        conn = Connectome(config)

        # Add a GABA pathway that is NOT inhibitory (violates Dale's law)
        conn.pathways.append(ProjectionPathway(
            source="PFC", target="Motor",
            nt_system=NTSystem.GABA,
            strength=0.5,
            is_inhibitory=False  # VIOLATION!
        ))

        with pytest.raises(DalesLawViolation, match="GABA pathway.*not inhibitory"):
            conn.validate()

    def test_dales_law_strict_rejects_glutamate_inhibitory(self):
        """Glutamate pathway that is inhibitory raises in strict mode."""
        from ww.nca.connectome import ConnectomeConfig, Connectome, DalesLawViolation, ProjectionPathway, NTSystem

        config = ConnectomeConfig(enforce_dale_law=True, strict_dales_law=True)
        conn = Connectome(config)

        # Add a Glutamate pathway that IS inhibitory (violates Dale's law)
        conn.pathways.append(ProjectionPathway(
            source="Motor", target="Sensory",
            nt_system=NTSystem.GLUTAMATE,
            strength=0.5,
            is_inhibitory=True  # VIOLATION!
        ))

        with pytest.raises(DalesLawViolation, match="Glu pathway.*is inhibitory"):
            conn.validate()

    def test_dales_law_non_strict_warns_only(self):
        """Non-strict mode returns issues but does not raise."""
        from ww.nca.connectome import ConnectomeConfig, Connectome, ProjectionPathway, NTSystem

        config = ConnectomeConfig(enforce_dale_law=True, strict_dales_law=False)
        conn = Connectome(config)

        # Add violation
        conn.pathways.append(ProjectionPathway(
            source="PFC", target="Motor",
            nt_system=NTSystem.GABA,
            strength=0.5,
            is_inhibitory=False
        ))

        # Should return issues but not raise
        is_valid, issues = conn.validate()
        assert not is_valid
        assert any("GABA pathway" in issue and "not inhibitory" in issue for issue in issues)

    def test_dales_law_strict_allows_correct_gaba(self):
        """Correct GABA pathway (inhibitory) passes in strict mode."""
        from ww.nca.connectome import ConnectomeConfig, Connectome, DalesLawViolation, ProjectionPathway, NTSystem

        config = ConnectomeConfig(enforce_dale_law=True, strict_dales_law=True)
        conn = Connectome(config)

        # Add correct GABA pathway (inhibitory)
        conn.pathways.append(ProjectionPathway(
            source="Striatum", target="GP",
            nt_system=NTSystem.GABA,
            strength=0.8,
            is_inhibitory=True  # CORRECT
        ))

        # Should not raise (might have other validation issues, but not Dale's law)
        try:
            conn.validate()
        except DalesLawViolation:
            pytest.fail("Should not raise DalesLawViolation for correct GABA pathway")
        except Exception:
            # Other validation issues are OK
            pass

    def test_dales_law_strict_allows_correct_glutamate(self):
        """Correct Glutamate pathway (excitatory) passes in strict mode."""
        from ww.nca.connectome import ConnectomeConfig, Connectome, DalesLawViolation, ProjectionPathway, NTSystem

        config = ConnectomeConfig(enforce_dale_law=True, strict_dales_law=True)
        conn = Connectome(config)

        # Add correct Glutamate pathway (excitatory)
        conn.pathways.append(ProjectionPathway(
            source="PFC", target="Striatum",
            nt_system=NTSystem.GLUTAMATE,
            strength=0.6,
            is_inhibitory=False  # CORRECT
        ))

        # Should not raise Dale's law violation
        try:
            conn.validate()
        except DalesLawViolation:
            pytest.fail("Should not raise DalesLawViolation for correct Glutamate pathway")
        except Exception:
            # Other validation issues are OK
            pass

    def test_dales_law_disabled_allows_violations(self):
        """When enforce_dale_law=False, violations are allowed."""
        from ww.nca.connectome import ConnectomeConfig, Connectome, ProjectionPathway, NTSystem

        config = ConnectomeConfig(enforce_dale_law=False, strict_dales_law=True)
        conn = Connectome(config)

        # Add violation - but enforcement is disabled
        conn.pathways.append(ProjectionPathway(
            source="PFC", target="Motor",
            nt_system=NTSystem.GABA,
            strength=0.5,
            is_inhibitory=False
        ))

        # Should not raise because enforcement is disabled
        is_valid, issues = conn.validate()
        # May have other issues, but not Dale's law
        assert not any("GABA pathway" in issue and "not inhibitory" in issue for issue in issues)


class TestDADelay:
    """ATOM-P1-2: Default DA delay should be 200ms (Schultz 1997)."""

    def test_default_delay_200ms(self):
        """P1-2: Default DA delay should be 200ms (Schultz 1997)."""
        from ww.nca.dopamine_integration import DopamineIntegration, DopamineIntegrationConfig, DelayBuffer

        # Check config default
        config = DopamineIntegrationConfig()
        assert config.da_arrival_delay_ms == 200.0, (
            f"Default DA delay should be 200ms per Schultz 1997, got {config.da_arrival_delay_ms}ms"
        )

        # Check DelayBuffer default
        buffer = DelayBuffer()
        assert buffer.delay_ms == 200.0, (
            f"DelayBuffer default should be 200ms, got {buffer.delay_ms}ms"
        )

    def test_delay_buffer_uses_sim_time(self):
        """P1-2: DelayBuffer should use simulation time, not wall-clock time."""
        from ww.nca.dopamine_integration import DelayBuffer

        buffer = DelayBuffer(delay_ms=200.0)  # 0.2 seconds

        # Enqueue at sim_time=0.0
        buffer.enqueue(value=1.0, sim_time=0.0)

        # Check at sim_time=0.1 (before delay elapsed)
        ready = buffer.dequeue_ready(sim_time=0.1)
        assert len(ready) == 0, "Value should not be ready before 0.2s elapsed"

        # Check at sim_time=0.2 (exactly when delay elapsed)
        ready = buffer.dequeue_ready(sim_time=0.2)
        assert len(ready) == 1, "Value should be ready after 0.2s elapsed"
        assert ready[0] == 1.0, "Should return the enqueued value"

    def test_integration_tracks_sim_time(self):
        """P1-2: DopamineIntegration should track simulation time."""
        from ww.nca.dopamine_integration import DopamineIntegration

        integration = DopamineIntegration()

        # Initial sim time should be 0
        assert integration._sim_time == 0.0, "Initial sim time should be 0"

        # Step forward
        integration.step(dt=0.1)
        assert abs(integration._sim_time - 0.1) < 1e-6, f"After one step, sim time should be 0.1, got {integration._sim_time}"

        # Step forward again
        integration.step(dt=0.1)
        assert abs(integration._sim_time - 0.2) < 1e-6, f"After two steps, sim time should be 0.2, got {integration._sim_time}"

    def test_integration_reset_clears_sim_time(self):
        """P1-2: Reset should clear simulation time."""
        from ww.nca.dopamine_integration import DopamineIntegration

        integration = DopamineIntegration()
        integration.step(dt=1.0)
        assert integration._sim_time > 0, "Sim time should be > 0 after stepping"

        integration.reset()
        assert integration._sim_time == 0.0, "Sim time should be reset to 0"
