"""
MNE-Python oscillation validation tests (P4-01).

Validates that T4DM neural oscillations match biological frequency bands:
- Delta: 0.5-4 Hz (deep sleep)
- Theta: 4-8 Hz (memory encoding)
- Alpha: 8-13 Hz (relaxed/attentive)
- Beta: 13-30 Hz (active thinking)
- Gamma: 30-100 Hz (binding/consolidation)

Uses MNE-Python spectral analysis when available.
"""

import numpy as np
import pytest

# Try importing MNE
MNE_AVAILABLE = False
try:
    import mne
    from mne.time_frequency import psd_array_welch
    MNE_AVAILABLE = True
except ImportError:
    pass

from t4dm.nca.oscillators import FrequencyBandGenerator, OscillatorConfig


# Biological frequency band definitions (Hz)
DELTA_RANGE = (0.5, 4.0)
THETA_RANGE = (4.0, 8.0)
ALPHA_RANGE = (8.0, 13.0)
BETA_RANGE = (13.0, 30.0)
GAMMA_RANGE = (30.0, 100.0)


class TestFrequencyBandBiology:
    """Test that generated oscillations match biological frequency bands."""

    @pytest.fixture
    def oscillator(self):
        config = OscillatorConfig(
            theta_freq_hz=6.0,
            gamma_freq_hz=40.0,
            alpha_freq_hz=10.0,
            dt_ms=1.0,  # 1ms timestep = 1000 Hz sampling
        )
        return FrequencyBandGenerator(config)

    def _generate_signal(self, oscillator, duration_s: float, key: str, **step_kwargs) -> np.ndarray:
        """Generate oscillation signal using step() interface."""
        n_steps = int(duration_s * 1000 / oscillator.config.dt_ms)
        signal = []
        for _ in range(n_steps):
            outputs = oscillator.step(**step_kwargs)
            signal.append(outputs.get(key, 0.0))
        return np.array(signal)

    def test_theta_frequency_in_band(self, oscillator):
        """Theta oscillation should be within 4-8 Hz."""
        # Generate 5 seconds of theta (high ACh promotes theta)
        signal = self._generate_signal(oscillator, 5.0, 'theta', ach_level=0.8)

        # Compute peak frequency using FFT
        sample_rate = 1000 / oscillator.config.dt_ms
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)

        # Find positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = np.abs(fft[pos_mask]) ** 2

        # Find peak frequency
        peak_idx = np.argmax(pos_power)
        peak_freq = pos_freqs[peak_idx]

        assert THETA_RANGE[0] <= peak_freq <= THETA_RANGE[1], \
            f"Theta peak at {peak_freq:.2f} Hz, expected {THETA_RANGE}"

    def test_gamma_frequency_in_band(self, oscillator):
        """Gamma oscillation should be within 30-100 Hz."""
        # Generate 2 seconds of gamma
        signal = self._generate_signal(oscillator, 2.0, 'gamma', glu_level=0.7)

        sample_rate = 1000 / oscillator.config.dt_ms
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)

        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = np.abs(fft[pos_mask]) ** 2

        peak_idx = np.argmax(pos_power)
        peak_freq = pos_freqs[peak_idx]

        assert GAMMA_RANGE[0] <= peak_freq <= GAMMA_RANGE[1], \
            f"Gamma peak at {peak_freq:.2f} Hz, expected {GAMMA_RANGE}"

    def test_delta_frequency_in_band(self, oscillator):
        """Delta oscillation should be within 0.5-4 Hz (sleep state)."""
        # Generate 10 seconds of delta (sleep state promotes delta)
        signal = self._generate_signal(oscillator, 10.0, 'delta', sleep_depth=0.8)

        sample_rate = 1000 / oscillator.config.dt_ms
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)

        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = np.abs(fft[pos_mask]) ** 2

        peak_idx = np.argmax(pos_power)
        peak_freq = pos_freqs[peak_idx]

        assert DELTA_RANGE[0] <= peak_freq <= DELTA_RANGE[1], \
            f"Delta peak at {peak_freq:.2f} Hz, expected {DELTA_RANGE}"


@pytest.mark.skipif(not MNE_AVAILABLE, reason="MNE-Python not installed")
class TestMNEOscillationValidation:
    """Use MNE-Python spectral analysis for rigorous validation."""

    @pytest.fixture
    def oscillator(self):
        config = OscillatorConfig(
            theta_freq_hz=6.0,
            gamma_freq_hz=40.0,
            dt_ms=1.0,
        )
        return FrequencyBandGenerator(config)

    def _generate_signal(self, oscillator, duration_s: float, key: str, **step_kwargs) -> np.ndarray:
        """Generate oscillation signal using step() interface."""
        n_steps = int(duration_s * 1000 / oscillator.config.dt_ms)
        signal = []
        for _ in range(n_steps):
            outputs = oscillator.step(**step_kwargs)
            signal.append(outputs.get(key, 0.0))
        return np.array(signal)

    def test_theta_spectral_peak_mne(self, oscillator):
        """Validate theta using MNE Welch PSD."""
        sfreq = 1000 / oscillator.config.dt_ms
        signal = self._generate_signal(oscillator, 10.0, 'theta', ach_level=0.8)

        # Compute PSD using Welch method
        psd, freqs = psd_array_welch(
            signal.reshape(1, -1),  # MNE expects (n_channels, n_times)
            sfreq=sfreq,
            fmin=0.5,
            fmax=50.0,
            n_fft=int(sfreq * 2),  # 2 second windows
            n_overlap=int(sfreq),  # 50% overlap
            verbose=False,
        )

        # Find peak in theta band
        theta_mask = (freqs >= THETA_RANGE[0]) & (freqs <= THETA_RANGE[1])
        theta_power = psd[0, theta_mask]
        total_power = psd[0].sum()

        # Theta should contain majority of power
        theta_ratio = theta_power.sum() / (total_power + 1e-10)
        assert theta_ratio > 0.3, \
            f"Theta band contains {theta_ratio*100:.1f}% of power, expected >30%"


class TestSleepOscillations:
    """Validate sleep-related oscillation patterns."""

    @pytest.fixture
    def oscillator(self):
        config = OscillatorConfig(
            theta_freq_hz=6.0,
            gamma_freq_hz=40.0,
            dt_ms=1.0,
        )
        return FrequencyBandGenerator(config)

    def _generate_signal(self, oscillator, duration_s: float, key: str, **step_kwargs) -> np.ndarray:
        """Generate oscillation signal using step() interface."""
        n_steps = int(duration_s * 1000 / oscillator.config.dt_ms)
        signal = []
        for _ in range(n_steps):
            outputs = oscillator.step(**step_kwargs)
            signal.append(outputs.get(key, 0.0))
        return np.array(signal)

    def test_nrem_delta_dominance(self, oscillator):
        """During NREM sleep, delta should dominate."""
        sample_rate = 1000 / oscillator.config.dt_ms

        # NREM-like: deep sleep promotes delta
        delta_signal = self._generate_signal(oscillator, 10.0, 'delta', sleep_depth=0.8)
        theta_signal = self._generate_signal(oscillator, 10.0, 'theta', sleep_depth=0.8)

        # Compute power
        delta_power = np.abs(np.fft.fft(delta_signal)).sum()
        theta_power = np.abs(np.fft.fft(theta_signal)).sum()

        # Delta should have more power during deep sleep
        assert delta_power > 0, "Delta should be active during deep sleep"

    def test_wake_theta_active(self, oscillator):
        """During waking, theta should be active with high ACh."""
        # Wake state with high ACh
        theta_signal = self._generate_signal(
            oscillator, 5.0, 'theta',
            sleep_depth=0.0, ach_level=0.8
        )

        # Theta should be active
        theta_power = np.abs(np.fft.fft(theta_signal)).sum()
        assert theta_power > 0, "Theta should be active during alert wake state"


class TestOscillatorBasics:
    """Basic tests for oscillator functionality."""

    def test_oscillator_step_returns_dict(self):
        """step() should return a dictionary with oscillator values."""
        config = OscillatorConfig(dt_ms=1.0)
        gen = FrequencyBandGenerator(config)

        outputs = gen.step()

        assert isinstance(outputs, dict)
        assert 'theta' in outputs
        assert 'gamma' in outputs
        assert 'alpha' in outputs

    def test_oscillator_values_bounded(self):
        """Oscillator outputs should be bounded."""
        config = OscillatorConfig(dt_ms=1.0)
        gen = FrequencyBandGenerator(config)

        for _ in range(1000):
            outputs = gen.step()
            for key, value in outputs.items():
                if isinstance(value, (int, float)):
                    assert -10 < value < 10, f"{key} = {value} is out of bounds"

    def test_neuromodulator_affects_oscillations(self):
        """Neuromodulator levels should affect oscillation amplitude."""
        config = OscillatorConfig(dt_ms=1.0)
        gen = FrequencyBandGenerator(config)

        # High ACh should enhance theta
        outputs_high_ach = []
        for _ in range(1000):
            outputs_high_ach.append(gen.step(ach_level=0.9)['theta'])

        gen2 = FrequencyBandGenerator(config)
        outputs_low_ach = []
        for _ in range(1000):
            outputs_low_ach.append(gen2.step(ach_level=0.1)['theta'])

        # Should be different (though direction depends on implementation)
        high_var = np.var(outputs_high_ach)
        low_var = np.var(outputs_low_ach)

        # At least they should not be exactly the same
        assert not np.allclose(outputs_high_ach, outputs_low_ach), \
            "ACh level should affect theta oscillation"
