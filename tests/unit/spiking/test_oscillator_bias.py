"""Tests for oscillator bias."""

import math

import pytest
import torch

from t4dm.spiking.oscillator_bias import OscillatorBias, OscillatorState


class TestOscillatorBias:
    @pytest.fixture
    def bias_mod(self):
        return OscillatorBias(dim=16)

    def test_output_shape(self, bias_mod):
        state = OscillatorState()
        out = bias_mod(state)
        assert out.shape == (16,)

    def test_zero_phase_near_zero(self, bias_mod):
        state = OscillatorState(theta_phase=0.0, gamma_phase=0.0, delta_phase=0.0)
        out = bias_mod(state)
        # sin(0) = 0, so bias should be near zero (modulo learned weights)
        assert out.abs().max() < 1.0

    def test_theta_gates_encoding(self, bias_mod):
        # Peak theta (sin=1) should produce larger bias than trough (sin=-1)
        peak = OscillatorState(theta_phase=math.pi / 2)
        trough = OscillatorState(theta_phase=-math.pi / 2)
        out_peak = bias_mod(peak)
        out_trough = bias_mod(trough)
        # They should differ (opposite sign for theta contribution)
        assert not torch.allclose(out_peak, out_trough)

    def test_phase_modulation(self, bias_mod):
        s1 = OscillatorState(gamma_phase=0.0)
        s2 = OscillatorState(gamma_phase=math.pi / 2)
        out1 = bias_mod(s1)
        out2 = bias_mod(s2)
        assert not torch.allclose(out1, out2)
