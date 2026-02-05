"""
Norse SNN Backend Wrapper for T4DM.

Provides hardware-efficient spiking neural network primitives using the Norse library.
This wrapper enables GPU-accelerated SNN computation while maintaining compatibility
with T4DM's existing spiking block architecture.

Norse (https://github.com/norse/norse) provides:
- Efficient LIF/LIFRecurrent implementations with CUDA support
- Pre-built surrogate gradient functions
- Optimized spike encoding/decoding
- Native PyTorch integration

This module wraps Norse to provide:
- Drop-in replacement for custom LIF neurons
- Unified interface for different neuron models
- State management compatible with cortical blocks
- Integration with T4DM neuromodulator systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from t4dm.learning.neuromodulators import NeuromodulatorState

logger = logging.getLogger(__name__)

# Check if Norse is available
try:
    import norse.torch as norse
    from norse.torch import LIFParameters, LIFState
    from norse.torch.functional.lif import lif_step

    NORSE_AVAILABLE = True
    logger.info("Norse library loaded successfully")
except ImportError:
    NORSE_AVAILABLE = False
    norse = None
    LIFParameters = None
    LIFState = None
    lif_step = None
    logger.warning("Norse library not available, falling back to custom implementation")


class NeuronModel(str, Enum):
    """Available neuron models in the SNN backend."""

    LIF = "lif"  # Leaky Integrate-and-Fire
    LIF_RECURRENT = "lif_recurrent"  # LIF with recurrent connections
    ADAPTIVE_LIF = "adaptive_lif"  # LIF with adaptation
    IZHIKEVICH = "izhikevich"  # Izhikevich neuron model
    CUSTOM = "custom"  # Custom implementation (fallback)


class SurrogateGradient(str, Enum):
    """Surrogate gradient functions for backpropagation through spikes."""

    SUPERSPIKE = "superspike"  # SuperSpike (Zenke & Ganguli 2018)
    TRIANGLE = "triangle"  # Triangular surrogate
    SIGMOID = "sigmoid"  # Sigmoid derivative
    ATAN = "atan"  # Arctangent (default in T4DM)
    BOXCAR = "boxcar"  # Boxcar function


@dataclass
class SNNConfig:
    """Configuration for the SNN backend."""

    neuron_model: NeuronModel = NeuronModel.LIF
    surrogate_gradient: SurrogateGradient = SurrogateGradient.SUPERSPIKE
    tau_mem: float = 10.0  # Membrane time constant (ms)
    tau_syn: float = 5.0  # Synaptic time constant (ms)
    v_th: float = 1.0  # Spike threshold
    v_reset: float = 0.0  # Reset potential
    v_leak: float = 0.0  # Leak potential
    alpha: float = 100.0  # Surrogate gradient sharpness
    dt: float = 1.0  # Time step (ms)
    use_norse: bool = True  # Use Norse if available
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SNNState:
    """State container for SNN neurons."""

    membrane: Tensor  # Membrane potentials
    synaptic: Tensor | None = None  # Synaptic currents (if applicable)
    adaptation: Tensor | None = None  # Adaptation variable (if adaptive)
    refractory: Tensor | None = None  # Refractory counter

    def detach(self) -> "SNNState":
        """Detach state from computation graph."""
        return SNNState(
            membrane=self.membrane.detach(),
            synaptic=self.synaptic.detach() if self.synaptic is not None else None,
            adaptation=self.adaptation.detach() if self.adaptation is not None else None,
            refractory=self.refractory.detach() if self.refractory is not None else None,
        )


class SNNBackend(nn.Module):
    """
    Unified SNN backend supporting Norse and custom implementations.

    This class provides a consistent interface for spiking neural networks,
    automatically using Norse when available for GPU acceleration, or
    falling back to custom implementations otherwise.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        config: SNNConfig | None = None,
    ):
        """
        Initialize the SNN backend.

        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension (number of neurons)
            config: SNN configuration
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.config = config or SNNConfig()

        # Determine if we should use Norse
        self.use_norse = self.config.use_norse and NORSE_AVAILABLE

        # Build the network
        if self.use_norse:
            self._build_norse_network()
        else:
            self._build_custom_network()

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        logger.info(
            f"SNNBackend initialized: size={hidden_size}, "
            f"model={self.config.neuron_model.value}, "
            f"norse={self.use_norse}"
        )

    def _build_norse_network(self) -> None:
        """Build network using Norse primitives."""
        # Compute alpha (leak factor) from time constant
        # alpha = exp(-dt / tau_mem)
        alpha = torch.exp(torch.tensor(-self.config.dt / self.config.tau_mem))

        # Create Norse LIF parameters
        self.lif_params = LIFParameters(
            tau_mem_inv=1.0 / self.config.tau_mem,
            tau_syn_inv=1.0 / self.config.tau_syn,
            v_th=torch.tensor(self.config.v_th),
            v_reset=torch.tensor(self.config.v_reset),
            v_leak=torch.tensor(self.config.v_leak),
            alpha=alpha.item(),
        )

        # Store for later use
        self._norse_alpha = alpha

    def _build_custom_network(self) -> None:
        """Build network using custom LIF implementation."""
        from t4dm.spiking.lif import LIFNeuron

        # Compute alpha from time constant
        alpha = float(torch.exp(torch.tensor(-self.config.dt / self.config.tau_mem)))

        self.lif_layer = LIFNeuron(
            size=self.hidden_size,
            alpha=alpha,
            v_thresh=self.config.v_th,
            beta=1.0,  # Full reset
            surrogate_alpha=self.config.alpha,
        )

    def init_state(self, batch_size: int) -> SNNState:
        """
        Initialize neuron state for a batch.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            Initialized SNNState
        """
        device = next(self.parameters()).device

        membrane = torch.zeros(batch_size, self.hidden_size, device=device)
        synaptic = torch.zeros(batch_size, self.hidden_size, device=device)

        return SNNState(membrane=membrane, synaptic=synaptic)

    def forward(
        self,
        x: Tensor,
        state: SNNState | None = None,
        neuromod_state: NeuromodulatorState | None = None,
    ) -> tuple[Tensor, SNNState]:
        """
        Forward pass through the SNN layer.

        Args:
            x: Input tensor (batch, input_size) or (batch, seq_len, input_size)
            state: Previous neuron state
            neuromod_state: Optional neuromodulator state for dynamic modulation

        Returns:
            Tuple of (spikes, new_state)
        """
        # Handle sequence input
        if x.dim() == 3:
            return self._forward_sequence(x, state, neuromod_state)

        # Project input
        current = self.input_proj(x)

        # Apply neuromodulator modulation if available
        if neuromod_state is not None:
            current = self._apply_neuromodulation(current, neuromod_state)

        # Initialize state if needed
        if state is None:
            state = self.init_state(x.size(0))

        # Run through LIF layer
        if self.use_norse:
            spikes, new_state = self._norse_forward(current, state)
        else:
            spikes, new_state = self._custom_forward(current, state)

        return spikes, new_state

    def _forward_sequence(
        self,
        x: Tensor,
        state: SNNState | None,
        neuromod_state: NeuromodulatorState | None,
    ) -> tuple[Tensor, SNNState]:
        """Process a sequence of inputs."""
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = self.init_state(batch_size)

        spikes_list = []

        for t in range(seq_len):
            spikes, state = self.forward(x[:, t], state, neuromod_state)
            spikes_list.append(spikes)

        spikes = torch.stack(spikes_list, dim=1)
        return spikes, state

    def _norse_forward(
        self, current: Tensor, state: SNNState
    ) -> tuple[Tensor, SNNState]:
        """Forward pass using Norse LIF."""
        # Convert to Norse state format
        norse_state = LIFState(
            z=state.membrane,  # Previous spikes (we use membrane for simplicity)
            v=state.membrane,
            i=state.synaptic if state.synaptic is not None else torch.zeros_like(state.membrane),
        )

        # Run Norse LIF step
        z, new_norse_state = lif_step(
            current,
            norse_state,
            self.lif_params,
        )

        # Convert back to our state format
        new_state = SNNState(
            membrane=new_norse_state.v,
            synaptic=new_norse_state.i,
        )

        return z, new_state

    def _custom_forward(
        self, current: Tensor, state: SNNState
    ) -> tuple[Tensor, SNNState]:
        """Forward pass using custom LIF."""
        spikes, membrane = self.lif_layer(current, state.membrane)

        new_state = SNNState(
            membrane=membrane,
            synaptic=state.synaptic,
        )

        return spikes, new_state

    def _apply_neuromodulation(
        self, current: Tensor, neuromod_state: NeuromodulatorState
    ) -> Tensor:
        """
        Apply neuromodulator modulation to input current.

        Biological basis:
        - ACh: Enhances signal-to-noise (multiplicative gain)
        - NE: Increases excitability (threshold modulation)
        - DA: Modulates learning (not directly applied here)
        """
        # ACh modulates gain (higher ACh = sharper responses)
        ach_gain = 0.8 + 0.4 * neuromod_state.acetylcholine_mode
        current = current * ach_gain

        # NE modulates baseline excitation (higher NE = more excitable)
        ne_baseline = 0.1 * (neuromod_state.norepinephrine_gain - 0.5)
        current = current + ne_baseline

        return current


class SpikeEncoder(nn.Module):
    """
    Encode continuous values into spike trains.

    Supports multiple encoding schemes:
    - Rate coding: spike probability proportional to value
    - Temporal coding: spike timing encodes value
    - Population coding: distributed representation across neurons
    """

    def __init__(
        self,
        encoding: str = "rate",
        num_steps: int = 10,
        gain: float = 1.0,
    ):
        """
        Initialize spike encoder.

        Args:
            encoding: Encoding scheme ('rate', 'temporal', 'population')
            num_steps: Number of time steps for encoding
            gain: Gain factor for encoding
        """
        super().__init__()
        self.encoding = encoding
        self.num_steps = num_steps
        self.gain = gain

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input values as spike trains.

        Args:
            x: Input tensor (batch, features)

        Returns:
            Spike trains (batch, num_steps, features)
        """
        batch_size = x.size(0)

        if self.encoding == "rate":
            # Rate coding: Poisson-like spike generation
            rates = torch.sigmoid(self.gain * x)  # Convert to [0, 1]
            rates = rates.unsqueeze(1).expand(-1, self.num_steps, -1)
            spikes = (torch.rand_like(rates) < rates).float()

        elif self.encoding == "temporal":
            # Temporal coding: Earlier spikes for higher values
            # Normalize x to [0, 1]
            x_norm = torch.sigmoid(self.gain * x)
            # Convert to spike times (higher value = earlier spike)
            spike_times = ((1 - x_norm) * self.num_steps).long()
            spike_times = spike_times.clamp(0, self.num_steps - 1)

            # Create spike train
            spikes = torch.zeros(batch_size, self.num_steps, x.size(-1), device=x.device)
            for t in range(self.num_steps):
                spikes[:, t] = (spike_times == t).float()

        elif self.encoding == "population":
            # Population coding: Gaussian tuning curves
            # Each neuron has a preferred value
            num_neurons = x.size(-1)
            preferred = torch.linspace(0, 1, num_neurons, device=x.device)
            x_norm = torch.sigmoid(self.gain * x)

            # Gaussian activation based on distance to preferred value
            sigma = 0.2
            activation = torch.exp(-0.5 * ((x_norm.unsqueeze(-1) - preferred) / sigma) ** 2)
            activation = activation.mean(dim=-1)  # Average across feature dimension

            # Generate spikes
            activation = activation.unsqueeze(1).expand(-1, self.num_steps, -1)
            spikes = (torch.rand_like(activation) < activation).float()

        else:
            raise ValueError(f"Unknown encoding scheme: {self.encoding}")

        return spikes


class SpikeDecoder(nn.Module):
    """
    Decode spike trains back to continuous values.

    Supports:
    - Rate decoding: Count spikes over time
    - Temporal decoding: Use first spike time
    - Weighted decoding: Learned weighted sum
    """

    def __init__(
        self,
        decoding: str = "rate",
        output_size: int | None = None,
        hidden_size: int | None = None,
    ):
        """
        Initialize spike decoder.

        Args:
            decoding: Decoding scheme ('rate', 'temporal', 'weighted')
            output_size: Output dimension (for weighted decoding)
            hidden_size: Hidden dimension (for weighted decoding)
        """
        super().__init__()
        self.decoding = decoding

        if decoding == "weighted" and output_size is not None and hidden_size is not None:
            self.decoder = nn.Linear(hidden_size, output_size)
        else:
            self.decoder = None

    def forward(self, spikes: Tensor) -> Tensor:
        """
        Decode spike trains to continuous values.

        Args:
            spikes: Spike trains (batch, num_steps, features)

        Returns:
            Decoded values (batch, features) or (batch, output_size)
        """
        if self.decoding == "rate":
            # Rate decoding: normalize spike count
            return spikes.mean(dim=1)

        elif self.decoding == "temporal":
            # Temporal decoding: use first spike time
            # Find first spike for each neuron
            first_spike = torch.argmax(spikes, dim=1).float()
            # Earlier spike = higher value
            num_steps = spikes.size(1)
            return 1.0 - (first_spike / num_steps)

        elif self.decoding == "weighted":
            # Weighted decoding: learned projection
            if self.decoder is None:
                raise ValueError("Weighted decoding requires output_size and hidden_size")
            # Sum spikes over time, then project
            spike_sum = spikes.sum(dim=1)
            return self.decoder(spike_sum)

        else:
            raise ValueError(f"Unknown decoding scheme: {self.decoding}")


class RecurrentSNNLayer(nn.Module):
    """
    Recurrent SNN layer with lateral connections.

    Implements recurrent dynamics where neurons can excite/inhibit
    each other, enabling winner-take-all and other competitive dynamics.
    """

    def __init__(
        self,
        size: int,
        config: SNNConfig | None = None,
        excitatory_ratio: float = 0.8,
    ):
        """
        Initialize recurrent SNN layer.

        Args:
            size: Number of neurons
            config: SNN configuration
            excitatory_ratio: Fraction of excitatory neurons (Dale's law)
        """
        super().__init__()
        self.size = size
        self.config = config or SNNConfig()
        self.excitatory_ratio = excitatory_ratio

        # Feedforward input
        self.snn = SNNBackend(size, size, config)

        # Recurrent connections (Dale's law compliant)
        num_exc = int(size * excitatory_ratio)
        self.register_buffer(
            "dale_mask",
            torch.cat([torch.ones(num_exc), -torch.ones(size - num_exc)]),
        )
        self.recurrent = nn.Linear(size, size, bias=False)

        # Initialize recurrent weights to be small
        nn.init.xavier_uniform_(self.recurrent.weight, gain=0.1)

    def forward(
        self,
        x: Tensor,
        state: SNNState | None = None,
        num_steps: int = 10,
    ) -> tuple[Tensor, SNNState]:
        """
        Forward pass with recurrent dynamics.

        Args:
            x: Input tensor (batch, size)
            state: Previous neuron state
            num_steps: Number of recurrent steps

        Returns:
            Tuple of (final_spikes, final_state)
        """
        if state is None:
            state = self.snn.init_state(x.size(0))

        spikes = torch.zeros_like(x)

        for _ in range(num_steps):
            # Apply Dale's law to recurrent weights
            recurrent_input = self.recurrent(spikes) * self.dale_mask

            # Combine feedforward and recurrent input
            total_input = x + recurrent_input

            # Project and run through SNN
            current = self.snn.input_proj(total_input)
            if self.snn.use_norse:
                spikes, state = self.snn._norse_forward(current, state)
            else:
                spikes, state = self.snn._custom_forward(current, state)

        return spikes, state


# Convenience function to check Norse availability
def is_norse_available() -> bool:
    """Check if Norse library is available."""
    return NORSE_AVAILABLE


# Factory function for creating SNN layers
def create_snn_layer(
    input_size: int,
    hidden_size: int,
    recurrent: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create an SNN layer.

    Args:
        input_size: Input dimension
        hidden_size: Hidden layer dimension
        recurrent: Whether to use recurrent connections
        **kwargs: Additional arguments for SNNConfig

    Returns:
        SNN layer module
    """
    config = SNNConfig(**kwargs)

    if recurrent:
        return RecurrentSNNLayer(hidden_size, config)
    else:
        return SNNBackend(input_size, hidden_size, config)


__all__ = [
    "SNNBackend",
    "SNNConfig",
    "SNNState",
    "SpikeEncoder",
    "SpikeDecoder",
    "RecurrentSNNLayer",
    "NeuronModel",
    "SurrogateGradient",
    "is_norse_available",
    "create_snn_layer",
    "NORSE_AVAILABLE",
]
