"""
Two-compartment dendritic neuron model.

Biological inspiration: L5 pyramidal neurons (Häusser & Mel, 2003)

The dendritic neuron has two compartments:
- Basal: Processes bottom-up sensory input
- Apical: Processes top-down contextual/predictive input

The coupling between compartments allows context to modulate
processing, enabling prediction error computation.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DendriticConfig:
    """Configuration for dendritic neurons."""
    input_dim: int = 1024        # BGE-M3 embedding dimension
    hidden_dim: int = 512        # Internal representation
    context_dim: int = 512       # Top-down context dimension
    coupling_strength: float = 0.5  # Basal-apical coupling
    tau_dendrite: float = 10.0   # Dendritic time constant (ms)
    tau_soma: float = 15.0       # Somatic time constant (ms)
    activation: str = "tanh"     # Activation function


class DendriticNeuron(nn.Module):
    """
    Two-compartment neuron model with basal and apical dendrites.

    Architecture:
    - Basal compartment: Processes bottom-up input (1024-dim BGE-M3 embedding)
    - Apical compartment: Processes top-down context (512-dim context vector)
    - Coupling: Modulates interaction strength (τ_dendrite < τ_soma)

    The mismatch between basal and apical activity serves as a
    prediction error signal for learning.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        context_dim: int = 512,
        coupling_strength: float = 0.5,
        tau_dendrite: float = 10.0,
        tau_soma: float = 15.0,
        activation: Literal["tanh", "relu", "gelu"] = "tanh"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.coupling_strength = coupling_strength
        self.tau_dendrite = tau_dendrite
        self.tau_soma = tau_soma

        # Validate time constants (dendrite should be faster)
        assert tau_dendrite < tau_soma, \
            f"tau_dendrite ({tau_dendrite}) must be < tau_soma ({tau_soma})"

        # Basal dendrite (bottom-up)
        self.W_basal = nn.Linear(input_dim, hidden_dim)

        # Apical dendrite (top-down context)
        self.W_apical = nn.Linear(context_dim, hidden_dim)

        # Context gating
        self.W_gate = nn.Linear(hidden_dim, hidden_dim)

        # Soma integration
        self.W_soma = nn.Linear(hidden_dim * 2, hidden_dim)

        # Activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in [self.W_basal, self.W_apical, self.W_gate, self.W_soma]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        basal_input: torch.Tensor,
        apical_input: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dendritic neuron.

        Args:
            basal_input: Bottom-up input (batch, input_dim)
            apical_input: Top-down context (batch, context_dim)
                         If None, uses zeros (no context)

        Returns:
            output: Integrated soma response (batch, hidden_dim)
            mismatch: Basal-apical difference (prediction error signal)
        """
        batch_size = basal_input.shape[0]

        # Default context if not provided
        if apical_input is None:
            apical_input = torch.zeros(
                batch_size, self.context_dim,
                device=basal_input.device,
                dtype=basal_input.dtype
            )

        # Basal processing (bottom-up)
        h_basal = self.activation(self.W_basal(basal_input))

        # Apical processing (top-down)
        h_apical = self.activation(self.W_apical(apical_input))

        # Context gating - modulates how much context influences output
        gate = torch.sigmoid(self.W_gate(h_apical))

        # Coupled representation with gating
        coupled = h_basal + self.coupling_strength * gate * h_apical

        # Soma integration
        soma_input = torch.cat([h_basal, coupled], dim=-1)
        output = self.activation(self.W_soma(soma_input))

        # Mismatch signal (prediction error)
        # Higher mismatch = context didn't predict input well
        mismatch = torch.norm(h_basal - h_apical, dim=-1)

        return output, mismatch

    def compute_context_influence(
        self,
        basal_input: torch.Tensor,
        apical_input: torch.Tensor
    ) -> float:
        """
        Compute how much context influences output.

        Returns ratio of output change with vs without context.
        """
        with torch.no_grad():
            out_with_context, _ = self.forward(basal_input, apical_input)
            out_without_context, _ = self.forward(basal_input, None)

            diff = torch.norm(out_with_context - out_without_context)
            baseline = torch.norm(out_without_context)

            return (diff / (baseline + 1e-8)).item()


class DendriticProcessor(nn.Module):
    """
    Multi-layer dendritic processing pipeline.

    Stacks multiple DendriticNeuron layers for hierarchical processing.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list[int] = [512, 256],
        context_dim: int = 512,
        coupling_strength: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.context_dim = context_dim

        # Build layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            self.layers.append(
                DendriticNeuron(
                    input_dim=dims[i],
                    hidden_dim=dims[i + 1],
                    context_dim=context_dim,
                    coupling_strength=coupling_strength
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Layer norm for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in hidden_dims
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through multi-layer processor.

        Args:
            x: Input tensor (batch, input_dim)
            context: Context vector (batch, context_dim)

        Returns:
            output: Final output (batch, hidden_dims[-1])
            mismatches: List of mismatch signals from each layer
        """
        mismatches = []

        for layer, norm in zip(self.layers, self.layer_norms):
            x, mismatch = layer(x, context)
            x = norm(x)
            x = self.dropout(x)
            mismatches.append(mismatch)

        return x, mismatches

    @property
    def output_dim(self) -> int:
        """Output dimension of processor."""
        return self.hidden_dims[-1]
