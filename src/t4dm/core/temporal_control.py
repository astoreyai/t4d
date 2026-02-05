"""
τ(t) Temporal Control Signal — gates memory writes, plasticity, and replay.

Enhanced version with dopamine modulation, spike timing integration,
and adaptive thresholds for memory encoding decisions.

τ(t) = σ(λ_ε·ε + λ_Δ·novelty + λ_r·reward + λ_da·dopamine + λ_θ·theta_phase)

Output ∈ (0,1) controls:
- Memory write decisions (high τ = allow write)
- Plasticity strength (τ modulates learning rate)
- Replay priority (high τ items replayed more)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from t4dm.learning.neuromodulators import NeuromodulatorState


class TemporalControlMode(str, Enum):
    """Operating modes for temporal control."""

    ENCODING = "encoding"  # θ phase 0-π, favor memory writes
    RETRIEVAL = "retrieval"  # θ phase π-2π, favor recall
    CONSOLIDATION = "consolidation"  # Sleep, replay mode
    MAINTENANCE = "maintenance"  # Low activity, housekeeping


@dataclass
class TemporalControlState:
    """State of the temporal control system."""

    tau: float  # Current gate value [0, 1]
    mode: TemporalControlMode  # Operating mode
    write_enabled: bool  # Whether writes are allowed
    plasticity_gain: float  # Multiplier for learning rate
    replay_priority: float  # Priority for replay selection
    spike_gating: float  # Gating for spike generation


class TemporalControlSignal(nn.Module):
    """
    Enhanced temporal control signal with dopamine modulation.

    Extends the basic τ(t) gate with:
    - Dopamine reward prediction error integration
    - Theta phase gating (encoding vs retrieval)
    - Adaptive threshold learning
    - Spike timing integration
    """

    def __init__(
        self,
        lambda_epsilon: float = 1.0,
        lambda_delta: float = 1.0,
        lambda_r: float = 1.0,
        lambda_da: float = 0.5,
        lambda_theta: float = 0.3,
        write_threshold: float = 0.5,
        plasticity_scale: float = 2.0,
    ):
        """
        Initialize the temporal control signal.

        Args:
            lambda_epsilon: Weight for prediction error signal
            lambda_delta: Weight for novelty signal
            lambda_r: Weight for reward signal
            lambda_da: Weight for dopamine modulation
            lambda_theta: Weight for theta phase modulation
            write_threshold: Threshold for allowing memory writes
            plasticity_scale: Maximum plasticity multiplier
        """
        super().__init__()

        # Learnable lambda weights
        self.lambdas = nn.Parameter(
            torch.tensor(
                [lambda_epsilon, lambda_delta, lambda_r, lambda_da, lambda_theta]
            )
        )

        # Adaptive threshold (learned)
        self.write_threshold = nn.Parameter(torch.tensor(write_threshold))
        self.plasticity_scale = plasticity_scale

        # Running statistics for normalization
        self.register_buffer("signal_mean", torch.zeros(5))
        self.register_buffer("signal_std", torch.ones(5))
        self.register_buffer("update_count", torch.tensor(0))

        # Mode-specific biases
        self.mode_biases = nn.ParameterDict(
            {
                "encoding": nn.Parameter(torch.tensor(0.2)),  # Favor writes
                "retrieval": nn.Parameter(torch.tensor(-0.2)),  # Favor recall
                "consolidation": nn.Parameter(torch.tensor(0.0)),  # Neutral
                "maintenance": nn.Parameter(torch.tensor(-0.3)),  # Reduce writes
            }
        )

    def forward(
        self,
        prediction_error: Tensor,
        novelty: Tensor,
        reward: Tensor,
        dopamine: Tensor | None = None,
        theta_phase: Tensor | None = None,
        mode: TemporalControlMode = TemporalControlMode.ENCODING,
    ) -> Tensor:
        """
        Compute the temporal control signal τ(t).

        Args:
            prediction_error: Prediction error magnitude |δ|
            novelty: Novelty score from surprise detection
            reward: Reward signal (positive or negative)
            dopamine: Dopamine level from RPE (optional)
            theta_phase: Theta oscillation phase [0, 2π] (optional)
            mode: Current operating mode

        Returns:
            Gate value τ(t) in (0, 1)
        """
        # Default dopamine to neutral if not provided
        if dopamine is None:
            dopamine = torch.zeros_like(prediction_error)

        # Default theta phase to encoding phase if not provided
        if theta_phase is None:
            theta_phase = torch.zeros_like(prediction_error)

        # Convert theta phase to encoding/retrieval bias
        # cos(θ) > 0 for θ ∈ [0, π/2) ∪ (3π/2, 2π] (encoding)
        # cos(θ) < 0 for θ ∈ (π/2, 3π/2) (retrieval)
        theta_bias = torch.cos(theta_phase)

        # Stack all signals
        signals = torch.stack(
            [prediction_error, novelty, reward, dopamine, theta_bias], dim=-1
        )

        # Update running statistics (for normalization)
        if self.training:
            self._update_statistics(signals)

        # Normalize signals for stable learning
        signals_norm = (signals - self.signal_mean) / (self.signal_std + 1e-6)

        # Compute weighted sum
        weighted_sum = (signals_norm * self.lambdas).sum(dim=-1)

        # Add mode-specific bias
        mode_bias = self.mode_biases[mode.value]
        weighted_sum = weighted_sum + mode_bias

        # Apply sigmoid activation
        tau = torch.sigmoid(weighted_sum)

        return tau

    def _update_statistics(self, signals: Tensor) -> None:
        """Update running mean and std for signal normalization."""
        batch_mean = signals.mean(dim=0)
        batch_std = signals.std(dim=0)

        # Exponential moving average
        momentum = 0.99
        self.signal_mean = momentum * self.signal_mean + (1 - momentum) * batch_mean
        self.signal_std = momentum * self.signal_std + (1 - momentum) * batch_std
        self.update_count += 1

    def compute_state(
        self,
        prediction_error: Tensor,
        novelty: Tensor,
        reward: Tensor,
        dopamine: Tensor | None = None,
        theta_phase: Tensor | None = None,
        mode: TemporalControlMode = TemporalControlMode.ENCODING,
    ) -> TemporalControlState:
        """
        Compute full temporal control state including derived values.

        Args:
            prediction_error: Prediction error magnitude
            novelty: Novelty score
            reward: Reward signal
            dopamine: Dopamine level (optional)
            theta_phase: Theta phase (optional)
            mode: Operating mode

        Returns:
            TemporalControlState with all derived values
        """
        tau = self.forward(
            prediction_error, novelty, reward, dopamine, theta_phase, mode
        )

        # Scalar conversion for state
        tau_val = tau.item() if tau.numel() == 1 else tau.mean().item()

        # Compute derived values
        write_enabled = tau_val > self.write_threshold.item()
        plasticity_gain = tau_val * self.plasticity_scale
        replay_priority = tau_val  # High τ = high priority for replay

        # Spike gating: reduce spike generation during low τ
        spike_gating = 0.5 + 0.5 * tau_val  # Range [0.5, 1.0]

        return TemporalControlState(
            tau=tau_val,
            mode=mode,
            write_enabled=write_enabled,
            plasticity_gain=plasticity_gain,
            replay_priority=replay_priority,
            spike_gating=spike_gating,
        )

    def from_neuromodulator_state(
        self,
        neuromod_state: NeuromodulatorState,
        prediction_error: float = 0.0,
        novelty: float = 0.0,
        theta_phase: float = 0.0,
    ) -> TemporalControlState:
        """
        Compute temporal control from neuromodulator state.

        Convenience method for integration with the neuromodulator orchestra.

        Args:
            neuromod_state: Current neuromodulator levels
            prediction_error: External prediction error signal
            novelty: External novelty signal
            theta_phase: Current theta oscillation phase

        Returns:
            TemporalControlState derived from neuromodulator dynamics
        """
        # Extract signals from neuromodulator state
        dopamine = neuromod_state.dopamine_rpe
        reward = neuromod_state.dopamine_rpe  # RPE is the reward signal
        ne_novelty = neuromod_state.norepinephrine_gain  # NE tracks novelty

        # Combine external novelty with NE-based novelty
        combined_novelty = 0.5 * novelty + 0.5 * ne_novelty

        # Determine mode from ACh level
        ach_mode = neuromod_state.acetylcholine_mode
        if ach_mode > 0.6:
            mode = TemporalControlMode.ENCODING
        elif ach_mode < 0.4:
            mode = TemporalControlMode.RETRIEVAL
        else:
            mode = TemporalControlMode.MAINTENANCE

        # Convert to tensors
        pe_t = torch.tensor(prediction_error)
        nov_t = torch.tensor(combined_novelty)
        rew_t = torch.tensor(reward)
        da_t = torch.tensor(dopamine)
        theta_t = torch.tensor(theta_phase)

        return self.compute_state(pe_t, nov_t, rew_t, da_t, theta_t, mode)

    def should_write(
        self,
        prediction_error: float,
        novelty: float,
        reward: float,
        dopamine: float = 0.0,
        theta_phase: float = 0.0,
        mode: TemporalControlMode = TemporalControlMode.ENCODING,
    ) -> bool:
        """
        Quick check if a memory should be written.

        Args:
            prediction_error: Prediction error magnitude
            novelty: Novelty score
            reward: Reward signal
            dopamine: Dopamine level
            theta_phase: Theta phase
            mode: Operating mode

        Returns:
            True if memory should be written
        """
        pe_t = torch.tensor(prediction_error)
        nov_t = torch.tensor(novelty)
        rew_t = torch.tensor(reward)
        da_t = torch.tensor(dopamine)
        theta_t = torch.tensor(theta_phase)

        tau = self.forward(pe_t, nov_t, rew_t, da_t, theta_t, mode)
        return tau.item() > self.write_threshold.item()

    def get_plasticity_multiplier(
        self,
        prediction_error: float,
        novelty: float,
        reward: float,
        dopamine: float = 0.0,
    ) -> float:
        """
        Get the plasticity multiplier for learning rate modulation.

        Args:
            prediction_error: Prediction error magnitude
            novelty: Novelty score
            reward: Reward signal
            dopamine: Dopamine level

        Returns:
            Multiplier for learning rate [0, plasticity_scale]
        """
        pe_t = torch.tensor(prediction_error)
        nov_t = torch.tensor(novelty)
        rew_t = torch.tensor(reward)
        da_t = torch.tensor(dopamine)

        tau = self.forward(pe_t, nov_t, rew_t, da_t)
        return tau.item() * self.plasticity_scale


# Backward compatibility alias
TemporalGate = TemporalControlSignal
