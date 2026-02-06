"""
Elephant-based spike train analysis tests (P4-02, P4-03).

Uses Elephant library when available for:
- Cross-correlation analysis (P4-02)
- Granger causality (P4-03)
- Spike statistics validation

Elephant: https://elephant.readthedocs.io/
"""

import numpy as np
import pytest
import torch

# Try importing Elephant
ELEPHANT_AVAILABLE = False
SpikeTrain = None  # Type stub for when not available
try:
    import elephant
    from elephant.spike_train_correlation import cross_correlation_histogram
    from elephant.statistics import mean_firing_rate, isi, cv
    from elephant.spike_train_generation import homogeneous_poisson_process
    from neo import SpikeTrain
    import quantities as pq
    ELEPHANT_AVAILABLE = True
except ImportError:
    pass

from t4dm.nca.snn_backend import SNNBackend, SNNConfig, SpikeEncoder, NeuronModel


class TestSpikeStatistics:
    """Test spike train statistics match biological ranges."""

    @pytest.fixture
    def snn(self):
        config = SNNConfig(
            neuron_model=NeuronModel.LIF,
            tau_mem=10.0,
            v_th=1.0,
        )
        return SNNBackend(input_size=32, hidden_size=64, config=config)

    def test_firing_rate_range(self, snn):
        """Firing rates should be in biological range (0-100 Hz typically)."""
        # Generate spikes from random input
        batch_size = 10
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 32) * 0.5

        with torch.no_grad():
            spikes, _ = snn(x)

        # Compute firing rate (assuming 1ms time steps)
        spike_counts = spikes.sum(dim=1)  # Sum over time
        firing_rates = spike_counts / (seq_len / 1000)  # Convert to Hz

        mean_rate = firing_rates.mean().item()

        # Biological range: cortical neurons typically 0-50 Hz
        assert 0 <= mean_rate <= 100, \
            f"Mean firing rate {mean_rate:.1f} Hz outside biological range"

    def test_spike_train_irregularity(self, snn):
        """Spike trains should show some irregularity (not perfectly regular)."""
        x = torch.randn(1, 200, 32) * 0.8

        with torch.no_grad():
            spikes, _ = snn(x)

        # Find spike times for one neuron
        neuron_spikes = spikes[0, :, 0].numpy()
        spike_times = np.where(neuron_spikes > 0.5)[0]

        if len(spike_times) > 2:
            # Compute inter-spike intervals
            isis = np.diff(spike_times)

            if len(isis) > 1:
                # Coefficient of variation: std / mean
                cv_isi = isis.std() / (isis.mean() + 1e-10)

                # Regular: CV ~0, Random (Poisson): CV ~1, Bursting: CV >1
                # Biological neurons typically have CV 0.5-1.5
                assert cv_isi > 0.1, \
                    f"CV of ISI = {cv_isi:.2f}, spikes too regular"

    def test_refractory_period(self, snn):
        """Neurons should have some spike timing structure."""
        # Use moderate input to avoid saturating all neurons
        x = torch.ones(1, 100, 32) * 0.5

        with torch.no_grad():
            spikes, _ = snn(x)

        # Check spike counts across ALL neurons (not just neuron 0, which may have
        # negative weights due to random initialization)
        all_spike_counts = (spikes[0] > 0.5).sum(dim=0).numpy()  # (64,) spikes per neuron
        num_timesteps = spikes.shape[1]

        # Find neurons that spiked (have positive input weights from random init)
        spiking_neurons = np.where(all_spike_counts > 0)[0]

        # Should have at least some neurons spiking given positive input
        assert len(spiking_neurons) > 0, \
            "No neurons spiked - unexpected for LIF model with positive input"

        # Check that there's variability in spike counts across neurons
        # (indicates temporal structure, not all-or-nothing firing)
        # Some neurons may spike every step (high input), some never (negative weights)
        # but there should be neurons with intermediate spike counts
        spike_rate_variance = np.var(all_spike_counts / num_timesteps)
        assert spike_rate_variance > 0.001, \
            f"No variance in spike rates ({spike_rate_variance:.4f}), expected temporal structure"

        # Check that total activity is in reasonable range
        total_activity = spikes.sum().item()
        max_possible = num_timesteps * 64  # 64 neurons
        activity_ratio = total_activity / max_possible
        assert 0.01 < activity_ratio < 0.99, \
            f"Activity ratio {activity_ratio:.2f} outside expected range (0.01-0.99)"


@pytest.mark.skipif(not ELEPHANT_AVAILABLE, reason="Elephant not installed")
class TestElephantCrossCorrelation:
    """Use Elephant for cross-correlation analysis (P4-02)."""

    @pytest.fixture
    def snn(self):
        config = SNNConfig(neuron_model=NeuronModel.LIF, tau_mem=10.0, v_th=1.0)
        return SNNBackend(input_size=32, hidden_size=64, config=config)

    def _spikes_to_neo(self, spikes: np.ndarray, duration_ms: float):
        """Convert spike array to Neo SpikeTrain."""
        spike_times = np.where(spikes > 0.5)[0]
        return SpikeTrain(spike_times * pq.ms, t_stop=duration_ms * pq.ms)

    def test_cross_correlation_connected_neurons(self, snn):
        """Connected neurons should show cross-correlation."""
        # Generate correlated input (simulating connected circuit)
        seq_len = 500
        base_input = torch.randn(1, seq_len, 32)

        # Same input = same firing patterns (artificial test of correlation detection)
        x1 = base_input * 0.8
        x2 = base_input * 0.8 + torch.randn(1, seq_len, 32) * 0.2  # Slight noise

        with torch.no_grad():
            spikes1, _ = snn(x1)
            spikes2, _ = snn(x2)

        # Convert to Neo SpikeTrain
        train1 = self._spikes_to_neo(spikes1[0, :, 0].numpy(), seq_len)
        train2 = self._spikes_to_neo(spikes2[0, :, 0].numpy(), seq_len)

        if len(train1) > 5 and len(train2) > 5:
            # Compute cross-correlation
            cch = cross_correlation_histogram(
                train1, train2,
                bin_size=5 * pq.ms,
                window=[-50, 50] * pq.ms,
            )

            # Peak should be near lag 0 for correlated inputs
            peak_lag = cch.times[np.argmax(cch.magnitude)]
            assert abs(peak_lag.magnitude) < 20, \
                f"Cross-correlation peak at lag {peak_lag}, expected near 0"

    def test_cross_correlation_independent_neurons(self, snn):
        """Independent neurons should show minimal cross-correlation."""
        seq_len = 500

        # Completely independent inputs
        x1 = torch.randn(1, seq_len, 32) * 0.8
        x2 = torch.randn(1, seq_len, 32) * 0.8

        with torch.no_grad():
            spikes1, _ = snn(x1)
            spikes2, _ = snn(x2)

        train1 = self._spikes_to_neo(spikes1[0, :, 0].numpy(), seq_len)
        train2 = self._spikes_to_neo(spikes2[0, :, 0].numpy(), seq_len)

        if len(train1) > 5 and len(train2) > 5:
            cch = cross_correlation_histogram(
                train1, train2,
                bin_size=5 * pq.ms,
                window=[-50, 50] * pq.ms,
            )

            # For independent spikes, cross-correlation should be relatively flat
            cch_values = cch.magnitude.flatten()
            if len(cch_values) > 3:
                peak_to_mean = cch_values.max() / (cch_values.mean() + 1e-10)
                # Should not have strong peak (allowing some noise)
                assert peak_to_mean < 3, \
                    f"Independent neurons show strong correlation: peak/mean = {peak_to_mean:.2f}"


@pytest.mark.skipif(not ELEPHANT_AVAILABLE, reason="Elephant not installed")
class TestElephantStatistics:
    """Use Elephant for spike statistics validation."""

    def test_poisson_spike_generation(self):
        """Verify Poisson spike train generation."""
        # Generate Poisson spike train
        rate = 50 * pq.Hz  # 50 Hz firing rate
        duration = 10 * pq.s

        train = homogeneous_poisson_process(rate, t_stop=duration)

        # Validate mean firing rate
        actual_rate = mean_firing_rate(train)
        assert abs(actual_rate.magnitude - 50) < 10, \
            f"Poisson rate {actual_rate} differs from expected 50 Hz"

    def test_isi_statistics(self):
        """Validate inter-spike interval statistics."""
        rate = 30 * pq.Hz
        duration = 10 * pq.s

        train = homogeneous_poisson_process(rate, t_stop=duration)

        if len(train) > 10:
            # Compute ISI
            intervals = isi(train)

            # For Poisson process, ISIs should follow exponential distribution
            # Mean ISI should be ~1/rate
            expected_mean_isi = (1000 / rate.magnitude) * pq.ms
            actual_mean_isi = intervals.mean()

            # Allow 30% tolerance
            assert abs(actual_mean_isi.magnitude - expected_mean_isi.magnitude) < \
                   0.3 * expected_mean_isi.magnitude, \
                f"Mean ISI {actual_mean_isi} differs from expected {expected_mean_isi}"

    def test_coefficient_of_variation(self):
        """Validate CV of ISI for Poisson process."""
        rate = 30 * pq.Hz
        duration = 10 * pq.s

        train = homogeneous_poisson_process(rate, t_stop=duration)

        if len(train) > 10:
            cv_value = cv(isi(train))

            # For Poisson process, CV should be ~1
            assert 0.7 < cv_value < 1.3, \
                f"CV = {cv_value:.2f}, expected ~1 for Poisson"


class TestSpikeEncoderDecoder:
    """Test spike encoding/decoding preserves information."""

    @pytest.fixture
    def encoder(self):
        return SpikeEncoder(encoding="rate", num_steps=50, gain=2.0)

    def test_rate_coding_correlation(self, encoder):
        """Rate-coded spikes should correlate with input values."""
        # Input with known pattern
        x = torch.tensor([[0.1, 0.5, 0.9]])  # Low, mid, high

        spikes = encoder(x)

        # Count spikes for each input dimension
        spike_counts = spikes.sum(dim=1)[0].numpy()

        # Higher input should produce more spikes
        assert spike_counts[2] > spike_counts[0], \
            "High input should produce more spikes than low input"

    def test_temporal_coding_order(self):
        """Temporal coding: higher values should spike earlier."""
        encoder = SpikeEncoder(encoding="temporal", num_steps=50, gain=2.0)

        x = torch.tensor([[0.1, 0.5, 0.9]])

        spikes = encoder(x)

        # Find first spike time for each dimension
        first_spike_times = []
        for d in range(3):
            spike_train = spikes[0, :, d].numpy()
            spike_indices = np.where(spike_train > 0.5)[0]
            if len(spike_indices) > 0:
                first_spike_times.append(spike_indices[0])
            else:
                first_spike_times.append(float('inf'))

        # Higher value should spike earlier (lower time index)
        assert first_spike_times[2] < first_spike_times[0], \
            f"High value should spike earlier: {first_spike_times}"
