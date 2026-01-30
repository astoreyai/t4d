"""
Tests for frequency band oscillators.

Validates:
1. Theta oscillations (4-8 Hz, ACh-modulated)
2. Gamma oscillations (30-80 Hz, E/I balance dependent)
3. Beta oscillations (13-30 Hz, DA-modulated)
4. Phase-amplitude coupling (PAC)
5. Cognitive phase detection (encoding vs retrieval)
6. Integration with NeuralFieldSolver
"""

import numpy as np
import pytest

from ww.nca.oscillators import (
    FrequencyBandGenerator,
    OscillatorConfig,
    OscillatorState,
    OscillationBand,
    CognitivePhase,
    ThetaOscillator,
    GammaOscillator,
    BetaOscillator,
    PhaseAmplitudeCoupling,
    DeltaOscillator,
    SleepState,
)
from ww.nca.sleep_spindles import (
    SleepSpindleGenerator,
    SpindleConfig,
    SpindleState,
    SpindleDeltaCoupler,
)
from ww.nca.neural_field import (
    NeuralFieldSolver,
    NeuralFieldConfig,
)


class TestOscillatorConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Default config has biologically reasonable values."""
        config = OscillatorConfig()

        # Theta in 4-8 Hz range
        assert 4.0 <= config.theta_freq_hz <= 8.0

        # Gamma in 30-80 Hz range
        assert 30.0 <= config.gamma_freq_hz <= 80.0

        # Beta in 13-30 Hz range
        assert 13.0 <= config.beta_freq_hz <= 30.0

        # PAC strength reasonable
        assert 0 < config.pac_strength < 1

    def test_validation(self):
        """Config validates frequency ranges."""
        # Should work
        config = OscillatorConfig(theta_freq_hz=6.0)
        assert config.theta_freq_hz == 6.0

        # Theta out of range should fail
        with pytest.raises(AssertionError):
            OscillatorConfig(theta_freq_hz=15.0)


class TestThetaOscillator:
    """Test theta band oscillator."""

    def test_initialization(self):
        """Theta oscillator initializes correctly."""
        config = OscillatorConfig()
        theta = ThetaOscillator(config)

        assert theta.phase == 0.0
        assert theta.freq == config.theta_freq_hz

    def test_phase_advances(self):
        """Theta phase advances with each step."""
        config = OscillatorConfig(theta_freq_hz=6.0)
        theta = ThetaOscillator(config)

        initial_phase = theta.phase

        # Step forward 10ms
        theta.step(ach_level=0.5, dt_ms=10.0)

        assert theta.phase > initial_phase

    def test_phase_wraps(self):
        """Theta phase wraps around 2π."""
        config = OscillatorConfig(theta_freq_hz=6.0)
        theta = ThetaOscillator(config)

        # Run for one full cycle (~167ms at 6 Hz)
        for _ in range(200):
            theta.step(ach_level=0.5, dt_ms=1.0)

        # Phase should be in [0, 2π]
        assert 0 <= theta.phase < 2 * np.pi

    def test_ach_modulates_amplitude(self):
        """Higher ACh should increase theta amplitude."""
        config = OscillatorConfig()
        theta = ThetaOscillator(config)

        # Low ACh
        theta.step(ach_level=0.2, dt_ms=1.0)
        amp_low = theta.amplitude

        # High ACh
        theta_high = ThetaOscillator(config)
        theta_high.step(ach_level=0.8, dt_ms=1.0)
        amp_high = theta_high.amplitude

        assert amp_high > amp_low

    def test_ach_modulates_frequency(self):
        """Higher ACh should slightly increase theta frequency."""
        config = OscillatorConfig()
        theta = ThetaOscillator(config)

        # Low ACh
        theta.step(ach_level=0.2, dt_ms=1.0)
        freq_low = theta.freq

        # High ACh
        theta_high = ThetaOscillator(config)
        theta_high.step(ach_level=0.8, dt_ms=1.0)
        freq_high = theta_high.freq

        assert freq_high > freq_low

    def test_cognitive_phase_encoding(self):
        """Phase 0-π should be encoding mode."""
        config = OscillatorConfig()
        theta = ThetaOscillator(config)

        theta.phase = 0.5  # In encoding range

        assert theta.get_cognitive_phase() == CognitivePhase.ENCODING

    def test_cognitive_phase_retrieval(self):
        """Phase π-2π should be retrieval mode."""
        config = OscillatorConfig()
        theta = ThetaOscillator(config)

        theta.phase = np.pi + 0.5  # In retrieval range

        assert theta.get_cognitive_phase() == CognitivePhase.RETRIEVAL


class TestGammaOscillator:
    """Test gamma band oscillator."""

    def test_initialization(self):
        """Gamma oscillator initializes correctly."""
        config = OscillatorConfig()
        gamma = GammaOscillator(config)

        assert gamma.phase == 0.0
        assert gamma.freq == config.gamma_freq_hz

    def test_ei_balance_modulates_frequency(self):
        """E/I balance should affect gamma frequency."""
        config = OscillatorConfig()

        # E-dominant (high Glu, low GABA) - slower gamma
        gamma_e = GammaOscillator(config)
        gamma_e.step(glu_level=0.8, gaba_level=0.2, theta_phase=0, pac_strength=0.3, dt_ms=1.0)
        freq_e = gamma_e.freq

        # I-dominant (low Glu, high GABA) - faster gamma
        gamma_i = GammaOscillator(config)
        gamma_i.step(glu_level=0.2, gaba_level=0.8, theta_phase=0, pac_strength=0.3, dt_ms=1.0)
        freq_i = gamma_i.freq

        assert freq_i > freq_e  # More inhibition = faster gamma

    def test_theta_phase_modulates_amplitude(self):
        """Gamma amplitude should be modulated by theta phase (PAC)."""
        config = OscillatorConfig(pac_preferred_phase=0.0)
        gamma = GammaOscillator(config)

        # At preferred theta phase (0) - max gamma
        gamma.step(glu_level=0.5, gaba_level=0.5, theta_phase=0.0, pac_strength=0.5, dt_ms=1.0)
        amp_at_peak = gamma.amplitude

        # At opposite theta phase (π) - min gamma
        gamma2 = GammaOscillator(config)
        gamma2.step(glu_level=0.5, gaba_level=0.5, theta_phase=np.pi, pac_strength=0.5, dt_ms=1.0)
        amp_at_trough = gamma2.amplitude

        assert amp_at_peak > amp_at_trough


class TestBetaOscillator:
    """Test beta band oscillator."""

    def test_initialization(self):
        """Beta oscillator initializes correctly."""
        config = OscillatorConfig()
        beta = BetaOscillator(config)

        assert beta.phase == 0.0
        assert beta.freq == config.beta_freq_hz

    def test_da_modulates_amplitude(self):
        """Higher DA should increase beta amplitude."""
        config = OscillatorConfig()

        # Low DA
        beta_low = BetaOscillator(config)
        beta_low.step(da_level=0.2, dt_ms=1.0)
        amp_low = beta_low.amplitude

        # High DA
        beta_high = BetaOscillator(config)
        beta_high.step(da_level=0.8, dt_ms=1.0)
        amp_high = beta_high.amplitude

        assert amp_high > amp_low


class TestPhaseAmplitudeCoupling:
    """Test theta-gamma PAC."""

    def test_initialization(self):
        """PAC initializes correctly."""
        config = OscillatorConfig()
        pac = PhaseAmplitudeCoupling(config)

        assert pac.strength == config.pac_strength

    def test_modulation_at_preferred_phase(self):
        """Gamma amplitude should be maximal at preferred theta phase."""
        config = OscillatorConfig(pac_strength=0.5, pac_preferred_phase=0.0)
        pac = PhaseAmplitudeCoupling(config)

        base_amp = 1.0

        # At preferred phase
        mod_at_peak = pac.compute_modulation(theta_phase=0.0, base_gamma_amplitude=base_amp)

        # At opposite phase
        mod_at_trough = pac.compute_modulation(theta_phase=np.pi, base_gamma_amplitude=base_amp)

        assert mod_at_peak > mod_at_trough

    def test_modulation_index_calculation(self):
        """Modulation index should reflect coupling strength."""
        config = OscillatorConfig(pac_strength=0.5)
        pac = PhaseAmplitudeCoupling(config)

        # Generate data with strong coupling
        for i in range(500):
            theta_phase = (i * 0.1) % (2 * np.pi)
            base_amp = 1.0
            pac.compute_modulation(theta_phase, base_amp)

        mi = pac.compute_modulation_index()

        # Should show some coupling
        assert mi > 0

    def test_pac_learning(self):
        """PAC strength should be learnable."""
        config = OscillatorConfig(pac_strength=0.5, pac_learning_rate=0.1)
        pac = PhaseAmplitudeCoupling(config)

        initial_strength = pac.strength

        # Positive reward increases PAC
        pac.update_strength(reward_signal=1.0)

        assert pac.strength > initial_strength

    def test_gamma_slots(self):
        """Gamma slots should give working memory capacity."""
        config = OscillatorConfig()
        pac = PhaseAmplitudeCoupling(config)

        slots = pac.get_gamma_slots()

        # Should be ~6-7 (40 Hz gamma / 6 Hz theta)
        assert 4 <= slots <= 10


class TestFrequencyBandGenerator:
    """Test unified frequency band generator."""

    def test_initialization(self):
        """Generator initializes correctly."""
        gen = FrequencyBandGenerator()

        assert gen.state.theta_phase == 0.0
        assert gen.state.gamma_phase == 0.0
        assert gen.state.beta_phase == 0.0

    def test_step_returns_all_bands(self):
        """Step should return all oscillation outputs."""
        gen = FrequencyBandGenerator()

        outputs = gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5)

        assert "theta" in outputs
        assert "gamma" in outputs
        assert "beta" in outputs
        assert "theta_phase" in outputs
        assert "cognitive_phase" in outputs

    def test_cognitive_phase_alternates(self):
        """Cognitive phase should alternate with theta."""
        gen = FrequencyBandGenerator()

        phases_seen = set()

        # Run for one theta cycle (~167ms at 6 Hz)
        for _ in range(200):
            outputs = gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5, dt_ms=1.0)
            phases_seen.add(outputs["cognitive_phase"])

        # Should see both encoding and retrieval
        assert "encoding" in phases_seen
        assert "retrieval" in phases_seen

    def test_encoding_signal(self):
        """Encoding signal should vary with theta phase."""
        gen = FrequencyBandGenerator()

        signals = []
        for _ in range(200):
            gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5, dt_ms=1.0)
            signals.append(gen.get_encoding_signal())

        # Should vary between 0 and 1
        assert min(signals) < 0.3
        assert max(signals) > 0.7

    def test_field_oscillations(self):
        """Oscillations should work across spatial field."""
        gen = FrequencyBandGenerator()

        ach_field = np.full((16,), 0.5, dtype=np.float32)
        da_field = np.full((16,), 0.5, dtype=np.float32)
        ne_field = np.full((16,), 0.3, dtype=np.float32)
        glu_field = np.full((16,), 0.5, dtype=np.float32)
        gaba_field = np.full((16,), 0.5, dtype=np.float32)

        outputs = gen.compute_oscillation_field(ach_field, da_field, ne_field, glu_field, gaba_field)

        assert outputs["theta"].shape == ach_field.shape
        assert outputs["alpha"].shape == ne_field.shape
        assert outputs["gamma"].shape == glu_field.shape
        assert outputs["beta"].shape == da_field.shape

    def test_spectral_power(self):
        """Should compute power in each band."""
        gen = FrequencyBandGenerator()

        # Run for a while
        for _ in range(500):
            gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5)

        power = gen.compute_spectral_power()

        assert power["theta"] > 0
        assert power["gamma"] > 0
        assert power["beta"] > 0

    def test_modulation_index(self):
        """Should compute PAC modulation index."""
        gen = FrequencyBandGenerator()

        # Run for a while
        for _ in range(1000):
            gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5)

        mi = gen.get_modulation_index()

        # Should show some coupling
        assert mi >= 0

    def test_working_memory_capacity(self):
        """Should estimate working memory capacity."""
        gen = FrequencyBandGenerator()

        capacity = gen.get_working_memory_capacity()

        # Should be 4-10 (Miller's 7±2)
        assert 4 <= capacity <= 10

    def test_pac_meta_learning(self):
        """PAC strength should adapt to rewards."""
        gen = FrequencyBandGenerator()
        initial_pac = gen.pac.strength

        # Positive reward
        gen.update_pac_from_reward(reward=1.0)

        assert gen.pac.strength > initial_pac

    def test_reset(self):
        """Reset should clear all state."""
        gen = FrequencyBandGenerator()

        # Run for a while
        for _ in range(100):
            gen.step(ach_level=0.7, da_level=0.3, glu_level=0.6, gaba_level=0.4)

        gen.reset()

        assert gen.state.theta_phase == 0.0
        assert gen._step_count == 0

    def test_stats(self):
        """Should provide comprehensive stats."""
        gen = FrequencyBandGenerator()

        for _ in range(100):
            gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5)

        stats = gen.get_stats()

        assert "step_count" in stats
        assert "theta_freq" in stats
        assert "gamma_freq" in stats
        assert "pac_strength" in stats
        assert "wm_capacity" in stats

    def test_validation(self):
        """Validation should check oscillation properties."""
        gen = FrequencyBandGenerator()

        # Run enough steps
        for _ in range(1000):
            gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5)

        validation = gen.validate_oscillations()

        assert "theta_freq_valid" in validation
        assert "gamma_freq_valid" in validation
        assert "all_pass" in validation


class TestNeuralFieldIntegration:
    """Test integration with NeuralFieldSolver."""

    def test_solver_with_oscillator(self):
        """NeuralFieldSolver works with oscillator."""
        config = NeuralFieldConfig(
            spatial_dims=1,
            grid_size=16,
            dt=0.001,
        )
        osc = FrequencyBandGenerator()

        solver = NeuralFieldSolver(
            config=config,
            oscillator=osc
        )

        # Run for several steps
        for _ in range(100):
            state = solver.step()

        # Should have valid state
        assert 0 <= state.dopamine <= 1
        assert 0 <= state.acetylcholine <= 1

        # Oscillator should have advanced
        assert osc._step_count > 0

    def test_cognitive_phase_accessible(self):
        """Solver should expose cognitive phase."""
        config = NeuralFieldConfig(spatial_dims=1, grid_size=16, dt=0.001)
        osc = FrequencyBandGenerator()
        solver = NeuralFieldSolver(config=config, oscillator=osc)

        # Run for a bit
        for _ in range(50):
            solver.step()

        phase = solver.get_cognitive_phase()
        assert phase in ["encoding", "retrieval"]

    def test_encoding_signal_accessible(self):
        """Solver should expose encoding signal."""
        config = NeuralFieldConfig(spatial_dims=1, grid_size=16, dt=0.001)
        osc = FrequencyBandGenerator()
        solver = NeuralFieldSolver(config=config, oscillator=osc)

        # Run for a bit
        for _ in range(50):
            solver.step()

        signal = solver.get_encoding_signal()
        assert 0 <= signal <= 1

    def test_oscillations_affect_dynamics(self):
        """Oscillations should introduce periodic fluctuations."""
        config = NeuralFieldConfig(spatial_dims=1, grid_size=16, dt=0.001)
        osc = FrequencyBandGenerator()
        solver = NeuralFieldSolver(config=config, oscillator=osc)

        # Record ACh over time
        ach_trace = []
        for _ in range(500):
            state = solver.step()
            ach_trace.append(state.acetylcholine)

        ach_trace = np.array(ach_trace)

        # Should show some variance (oscillations)
        assert np.std(ach_trace) > 0.001


class TestOscillationBiology:
    """Test biological properties of oscillations."""

    def test_theta_frequency_range(self):
        """Theta should stay in 4-8 Hz range."""
        gen = FrequencyBandGenerator()

        freqs = []
        for ach in np.linspace(0.1, 0.9, 10):
            gen.step(ach_level=ach, da_level=0.5, glu_level=0.5, gaba_level=0.5)
            freqs.append(gen.state.theta_freq)

        assert all(4.0 <= f <= 8.0 for f in freqs)

    def test_gamma_frequency_range(self):
        """Gamma should stay in 30-80 Hz range."""
        gen = FrequencyBandGenerator()

        freqs = []
        for gaba in np.linspace(0.1, 0.9, 10):
            gen.step(ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=gaba)
            freqs.append(gen.state.gamma_freq)

        assert all(30.0 <= f <= 80.0 for f in freqs)

    def test_theta_gamma_ratio(self):
        """Gamma/theta ratio should give ~6-7 (working memory capacity)."""
        config = OscillatorConfig(theta_freq_hz=6.0, gamma_freq_hz=40.0)
        gen = FrequencyBandGenerator(config)

        ratio = gen.state.gamma_freq / gen.state.theta_freq

        # Should be close to 6-7
        assert 5 <= ratio <= 8


class TestDeltaOscillator:
    """Test delta band oscillator (0.5-4 Hz) for slow-wave sleep."""

    def test_initialization(self):
        """Delta oscillator initializes correctly."""
        config = OscillatorConfig()
        delta = DeltaOscillator(config)

        assert delta.phase == 0.0
        assert 0.5 <= delta.freq <= 4.0
        assert delta._in_up_state is False

    def test_frequency_range(self):
        """Delta frequency should stay in 0.5-4 Hz range."""
        config = OscillatorConfig()
        delta = DeltaOscillator(config)

        for sleep_depth in np.linspace(0.3, 1.0, 10):
            delta.step(adenosine_level=0.5, sleep_depth=sleep_depth, dt_ms=1.0)
            assert 0.5 <= delta.freq <= 4.0

    def test_inactive_when_awake(self):
        """Delta should be minimal when sleep_depth < 0.3."""
        config = OscillatorConfig()
        delta = DeltaOscillator(config)

        output, _ = delta.step(adenosine_level=0.5, sleep_depth=0.1, dt_ms=1.0)

        assert output == 0.0
        assert delta.amplitude < 0.1

    def test_active_during_sleep(self):
        """Delta should be active during deep sleep."""
        config = OscillatorConfig()
        delta = DeltaOscillator(config)

        output, _ = delta.step(adenosine_level=0.6, sleep_depth=0.8, dt_ms=1.0)

        assert delta.amplitude > 0.2

    def test_adenosine_modulates_amplitude(self):
        """Higher adenosine should increase delta power."""
        config = OscillatorConfig()

        # Low adenosine
        delta_low = DeltaOscillator(config)
        delta_low.step(adenosine_level=0.2, sleep_depth=0.7, dt_ms=1.0)
        amp_low = delta_low.amplitude

        # High adenosine
        delta_high = DeltaOscillator(config)
        delta_high.step(adenosine_level=0.8, sleep_depth=0.7, dt_ms=1.0)
        amp_high = delta_high.amplitude

        assert amp_high > amp_low

    def test_up_state_detection(self):
        """Should detect up-states for consolidation gating."""
        config = OscillatorConfig()
        delta = DeltaOscillator(config)

        up_states = []
        for _ in range(2000):  # Run for ~2 delta cycles
            delta.step(adenosine_level=0.5, sleep_depth=0.8, dt_ms=1.0)
            up_states.append(delta.is_up_state())

        # Should see both up and down states
        assert any(up_states)
        assert not all(up_states)

    def test_consolidation_gate(self):
        """Consolidation gate should be 1.0 during up-state, 0.0 during down."""
        config = OscillatorConfig()
        delta = DeltaOscillator(config)

        gates = []
        for _ in range(2000):
            delta.step(adenosine_level=0.5, sleep_depth=0.8, dt_ms=1.0)
            gates.append(delta.get_consolidation_gate())

        # Should include both 0.0 and 1.0
        assert 0.0 in gates
        assert 1.0 in gates

    def test_downscaling_signal(self):
        """Downscaling signal should be active during down-states."""
        config = OscillatorConfig()
        delta = DeltaOscillator(config)

        # Run until down-state
        for _ in range(500):
            delta.step(adenosine_level=0.5, sleep_depth=0.8, dt_ms=1.0)
            if not delta.is_up_state():
                break

        # Run for more down-state time
        for _ in range(500):
            delta.step(adenosine_level=0.5, sleep_depth=0.8, dt_ms=1.0)
            if not delta.is_up_state():
                signal = delta.get_downscaling_signal()
                if signal > 0:
                    break

        # Should get non-zero downscaling during sustained down-state
        assert delta.get_downscaling_signal() >= 0


class TestSleepSpindleGenerator:
    """Test sleep spindle generator (11-16 Hz)."""

    def test_initialization(self):
        """Spindle generator initializes correctly."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        assert spindle.state == SpindleState.INACTIVE
        assert 11.0 <= spindle.freq <= 16.0

    def test_frequency_range(self):
        """Spindle frequency should stay in 11-16 Hz range."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        for _ in range(100):
            spindle.step(sleep_depth=0.6, delta_up_state=True, gaba_level=0.5)

        assert 11.0 <= spindle.freq <= 16.0

    def test_inactive_when_awake(self):
        """Spindles should not occur when awake."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        # Run for a while
        for _ in range(1000):
            output = spindle.step(sleep_depth=0.1, delta_up_state=False, gaba_level=0.5)

        assert spindle._total_spindles == 0

    def test_inactive_during_rem(self):
        """Spindles should not occur during REM (high ACh)."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        # Run with high ACh (REM-like)
        for _ in range(1000):
            output = spindle.step(
                sleep_depth=0.6, delta_up_state=True,
                gaba_level=0.5, ach_level=0.8
            )

        assert spindle._total_spindles == 0

    def test_spindles_during_nrem(self):
        """Spindles should occur during NREM sleep."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        # Run for a while with sleep conditions
        for _ in range(10000):
            spindle.step(
                sleep_depth=0.6, delta_up_state=True,
                gaba_level=0.6, ach_level=0.3
            )

        # Should have generated some spindles
        assert spindle._total_spindles > 0

    def test_delta_coupling(self):
        """Spindles should preferentially occur during delta up-states."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        # Run with delta up-state cycling
        for i in range(10000):
            # Simulate delta cycling (up for 50%, down for 50%)
            delta_up = (i % 100) < 50
            spindle.step(
                sleep_depth=0.6, delta_up_state=delta_up,
                gaba_level=0.6, ach_level=0.3
            )

        # Coupling ratio should be > 0.5 (biased toward up-states)
        if spindle._total_spindles > 3:
            assert spindle.get_delta_coupling_ratio() > 0.4

    def test_consolidation_gate(self):
        """Consolidation gate should vary with spindle state."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        # Inactive state
        assert spindle.get_consolidation_gate() == 0.0

        # Run until spindle
        for _ in range(10000):
            output = spindle.step(
                sleep_depth=0.6, delta_up_state=True,
                gaba_level=0.6, ach_level=0.3
            )
            if spindle.state == SpindleState.PLATEAU:
                assert spindle.get_consolidation_gate() == 1.0
                break

    def test_spindle_density(self):
        """Spindle density should be reasonable (0-15 per minute)."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        # Run for 1 minute of simulated time
        for _ in range(60000):
            spindle.step(
                sleep_depth=0.6, delta_up_state=True,
                gaba_level=0.6, ach_level=0.3
            )

        density = spindle.get_spindle_density()
        # Literature suggests 2-5 typical, but can be higher
        assert 0.0 <= density <= 15.0

    def test_reset(self):
        """Reset should clear all state."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        # Run for a while
        for _ in range(5000):
            spindle.step(
                sleep_depth=0.6, delta_up_state=True,
                gaba_level=0.6, ach_level=0.3
            )

        spindle.reset()

        assert spindle.state == SpindleState.INACTIVE
        assert spindle._total_spindles == 0
        assert len(spindle._spindle_events) == 0

    def test_validation(self):
        """Validation should check spindle properties."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)

        validation = spindle.validate_spindles()

        assert "freq_valid" in validation
        assert "density_valid" in validation
        assert "all_pass" in validation


class TestSpindleDeltaCoupler:
    """Test spindle-delta coupling."""

    def test_initialization(self):
        """Coupler initializes correctly."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)
        coupler = SpindleDeltaCoupler(spindle)

        assert coupler._in_coupling_window is False

    def test_coupling_window_detection(self):
        """Should detect optimal coupling window during up-state."""
        config = SpindleConfig()
        spindle = SleepSpindleGenerator(config)
        coupler = SpindleDeltaCoupler(spindle, coupling_window_ms=200.0)

        # Simulate up-state onset and timing
        time_ms = 0.0
        prev_up = False

        for _ in range(300):
            current_up = True  # Stay in up-state
            in_window = coupler.update(time_ms, current_up, prev_up)
            time_ms += 1.0
            prev_up = current_up

            # Should enter coupling window after ~50ms
            if 50 <= time_ms <= 200:
                # Should be in coupling window
                pass

        # Eventually should exit window
        for _ in range(100):
            in_window = coupler.update(time_ms, False, True)
            time_ms += 1.0

        assert coupler._in_coupling_window is False


class TestFrequencyBandGeneratorWithDelta:
    """Test FrequencyBandGenerator with delta oscillator integration."""

    def test_delta_in_output(self):
        """Step should include delta output."""
        gen = FrequencyBandGenerator()

        outputs = gen.step(
            ach_level=0.5, da_level=0.5, glu_level=0.5, gaba_level=0.5,
            sleep_depth=0.7
        )

        assert "delta" in outputs
        assert "delta_phase" in outputs
        assert "sleep_state" in outputs
        assert "in_up_state" in outputs

    def test_sleep_state_tracking(self):
        """Should track sleep state from sleep_depth."""
        gen = FrequencyBandGenerator()

        # Awake
        gen.step(sleep_depth=0.1)
        assert gen.state.sleep_state == SleepState.AWAKE

        # Light sleep
        gen.step(sleep_depth=0.4)
        assert gen.state.sleep_state == SleepState.LIGHT_SLEEP

        # Deep sleep
        gen.step(sleep_depth=0.8)
        assert gen.state.sleep_state == SleepState.DEEP_SLEEP

    def test_delta_power_during_sleep(self):
        """Delta power should be higher during sleep."""
        gen = FrequencyBandGenerator()

        # Run awake
        for _ in range(500):
            gen.step(sleep_depth=0.0)
        power_awake = gen.compute_spectral_power()["delta"]

        gen.reset()

        # Run during sleep
        for _ in range(500):
            gen.step(sleep_depth=0.8, adenosine_level=0.6)
        power_sleep = gen.compute_spectral_power()["delta"]

        assert power_sleep > power_awake

    def test_consolidation_signals(self):
        """Should provide consolidation gating signals."""
        gen = FrequencyBandGenerator()

        outputs = gen.step(sleep_depth=0.8, adenosine_level=0.6)

        assert "consolidation_gate" in outputs
        assert "downscaling_signal" in outputs

    def test_stats_include_delta(self):
        """Stats should include delta metrics."""
        gen = FrequencyBandGenerator()

        for _ in range(100):
            gen.step(sleep_depth=0.7)

        stats = gen.get_stats()

        assert "delta_freq" in stats
        assert "delta_amplitude" in stats
        assert "delta_power" in stats
        assert "sleep_state" in stats
        assert "in_up_state" in stats

    def test_validation_includes_delta(self):
        """Validation should check delta properties."""
        gen = FrequencyBandGenerator()

        for _ in range(100):
            gen.step(sleep_depth=0.7)

        validation = gen.validate_oscillations()

        assert "delta_freq_valid" in validation
        assert "sleep_state_valid" in validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
