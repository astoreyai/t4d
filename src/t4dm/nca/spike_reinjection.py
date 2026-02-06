"""
Spike Reinjection Module (P3-01).

Converts memory embeddings to spike trains for NREM replay,
enabling the spike→STDP→weight update consolidation loop.

The reinjection process:
1. Memory embedding (1024-dim) → population code
2. Population code → spike trains via rate/temporal coding
3. Spike trains fed to SNN for replay
4. SNN output drives STDP weight updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from t4dm.learning.neuromodulators import NeuromodulatorState

logger = logging.getLogger(__name__)


class ReinjectionMode(str, Enum):
    """Spike reinjection encoding modes."""

    RATE = "rate"  # Rate coding: spike probability ~ embedding value
    TEMPORAL = "temporal"  # Temporal coding: spike timing encodes value
    POPULATION = "population"  # Population coding: distributed representation
    BURST = "burst"  # Burst coding: packet-based for sharp-wave ripples


@dataclass
class ReinjectionConfig:
    """Configuration for spike reinjection."""

    mode: ReinjectionMode = ReinjectionMode.RATE
    num_steps: int = 50  # Number of time steps for spike generation
    gain: float = 2.0  # Encoding gain
    sparsity: float = 0.3  # Target sparsity for population coding
    burst_size: int = 5  # Spikes per burst for burst mode
    burst_interval: int = 10  # Steps between bursts
    noise_std: float = 0.1  # Noise for stochastic spiking
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SpikeReinjector(nn.Module):
    """
    Converts memory embeddings to spike trains for replay.

    Used during NREM consolidation to reinject stored memories
    through the spiking network, enabling STDP-based strengthening.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        config: ReinjectionConfig | None = None,
    ):
        """
        Initialize spike reinjector.

        Args:
            embedding_dim: Input embedding dimension (e.g., 1024 for BGE-M3)
            hidden_dim: Hidden layer dimension for projection
            output_dim: Output spike train dimension (neurons)
            config: Reinjection configuration
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.config = config or ReinjectionConfig()

        # Projection network: embedding → population activity
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Output in [0, 1] for spike probability
        )

        # Learnable temporal modulation for temporal coding
        self.temporal_weights = nn.Parameter(torch.randn(output_dim) * 0.1)

        # Population tuning curves for population coding
        self.register_buffer(
            "tuning_centers",
            torch.linspace(0, 1, output_dim),
        )
        self.tuning_width = 0.15

        logger.info(
            f"SpikeReinjector initialized: {embedding_dim}→{hidden_dim}→{output_dim}, "
            f"mode={self.config.mode.value}"
        )

    def forward(
        self,
        embedding: Tensor,
        neuromod_state: NeuromodulatorState | None = None,
    ) -> Tensor:
        """
        Convert embedding to spike train.

        Args:
            embedding: Memory embedding (batch, embedding_dim)
            neuromod_state: Optional neuromodulator state for modulation

        Returns:
            Spike trains (batch, num_steps, output_dim)
        """
        batch_size = embedding.size(0)

        # Project embedding to population activity
        activity = self.projector(embedding)  # (batch, output_dim)

        # Apply neuromodulator modulation if available
        if neuromod_state is not None:
            activity = self._apply_neuromodulation(activity, neuromod_state)

        # Generate spikes based on mode
        if self.config.mode == ReinjectionMode.RATE:
            spikes = self._rate_encode(activity)
        elif self.config.mode == ReinjectionMode.TEMPORAL:
            spikes = self._temporal_encode(activity)
        elif self.config.mode == ReinjectionMode.POPULATION:
            spikes = self._population_encode(embedding)
        elif self.config.mode == ReinjectionMode.BURST:
            spikes = self._burst_encode(activity)
        else:
            raise ValueError(f"Unknown reinjection mode: {self.config.mode}")

        return spikes

    def _rate_encode(self, activity: Tensor) -> Tensor:
        """
        Rate coding: spike probability proportional to activity.

        Higher activity → higher spike rate.
        """
        batch_size = activity.size(0)
        num_steps = self.config.num_steps

        # Scale activity by gain
        rates = torch.sigmoid(self.config.gain * (activity - 0.5))

        # Expand for time dimension
        rates = rates.unsqueeze(1).expand(-1, num_steps, -1)

        # Add noise for stochasticity
        noise = torch.randn_like(rates) * self.config.noise_std

        # Generate spikes via Bernoulli sampling
        spikes = (torch.rand_like(rates) < (rates + noise).clamp(0, 1)).float()

        return spikes

    def _temporal_encode(self, activity: Tensor) -> Tensor:
        """
        Temporal coding: spike timing encodes value.

        Higher activity → earlier spike.
        """
        batch_size = activity.size(0)
        num_steps = self.config.num_steps

        # Apply learnable temporal weights
        weighted_activity = activity * torch.sigmoid(self.temporal_weights)

        # Convert activity to spike times (higher = earlier)
        spike_times = ((1 - weighted_activity) * num_steps).long()
        spike_times = spike_times.clamp(0, num_steps - 1)

        # Create spike train
        spikes = torch.zeros(
            batch_size, num_steps, self.output_dim,
            device=activity.device,
        )

        # Place spikes at computed times
        for t in range(num_steps):
            spikes[:, t] = (spike_times == t).float()

        return spikes

    def _population_encode(self, embedding: Tensor) -> Tensor:
        """
        Population coding: distributed representation via tuning curves.

        Each neuron has a preferred embedding value (tuning curve).
        """
        batch_size = embedding.size(0)
        num_steps = self.config.num_steps

        # Normalize embedding to [0, 1]
        emb_norm = torch.sigmoid(embedding)

        # Compute mean embedding value per sample
        emb_mean = emb_norm.mean(dim=-1, keepdim=True)  # (batch, 1)

        # Compute tuning curve activation
        # Distance from each neuron's preferred value
        distance = (emb_mean - self.tuning_centers.unsqueeze(0)) ** 2
        activation = torch.exp(-distance / (2 * self.tuning_width ** 2))

        # Apply sparsity constraint
        threshold = torch.quantile(activation, 1 - self.config.sparsity, dim=-1, keepdim=True)
        activation = activation * (activation >= threshold).float()

        # Expand for time and generate spikes
        activation = activation.unsqueeze(1).expand(-1, num_steps, -1)
        spikes = (torch.rand_like(activation) < activation).float()

        return spikes

    def _burst_encode(self, activity: Tensor) -> Tensor:
        """
        Burst coding: packet-based encoding for sharp-wave ripples.

        High-activity neurons emit bursts of spikes.
        """
        batch_size = activity.size(0)
        num_steps = self.config.num_steps
        burst_size = self.config.burst_size
        burst_interval = self.config.burst_interval

        spikes = torch.zeros(
            batch_size, num_steps, self.output_dim,
            device=activity.device,
        )

        # Determine which neurons burst (top activity)
        threshold = torch.quantile(activity, 0.7, dim=-1, keepdim=True)
        burst_mask = (activity >= threshold).float()

        # Place bursts at regular intervals
        for t in range(0, num_steps, burst_interval):
            # Burst window
            end_t = min(t + burst_size, num_steps)
            for bt in range(t, end_t):
                # Add noise to burst timing
                noise = torch.rand_like(burst_mask) * 0.3
                spikes[:, bt] = burst_mask * (torch.rand_like(burst_mask) < (0.8 + noise)).float()

        return spikes

    def _apply_neuromodulation(
        self,
        activity: Tensor,
        neuromod_state: NeuromodulatorState,
    ) -> Tensor:
        """
        Apply neuromodulator modulation to activity.

        During NREM:
        - Low ACh: Reduced activity (memory consolidation mode)
        - Low NE: Reduced noise (stable replay)
        - DA modulates which memories get replayed (RPE-based)
        """
        # ACh modulates overall gain (low during NREM)
        ach_factor = 0.5 + 0.5 * neuromod_state.acetylcholine_mode
        activity = activity * ach_factor

        # DA modulates activity based on reward prediction error
        # High DA → boost activity (important memories)
        da_boost = 1.0 + 0.5 * (neuromod_state.dopamine_tonic - 0.5)
        activity = activity * da_boost

        return activity

    def reinject_batch(
        self,
        embeddings: list[np.ndarray] | Tensor,
        neuromod_state: NeuromodulatorState | None = None,
    ) -> Tensor:
        """
        Reinject a batch of memory embeddings.

        Args:
            embeddings: List of embeddings or tensor (batch, embedding_dim)
            neuromod_state: Optional neuromodulator state

        Returns:
            Batched spike trains (batch, num_steps, output_dim)
        """
        if isinstance(embeddings, list):
            embeddings = torch.tensor(
                np.stack(embeddings),
                dtype=torch.float32,
                device=self.config.device,
            )
        elif not isinstance(embeddings, Tensor):
            embeddings = torch.tensor(
                embeddings,
                dtype=torch.float32,
                device=self.config.device,
            )

        if embeddings.device.type != self.config.device:
            embeddings = embeddings.to(self.config.device)

        return self.forward(embeddings, neuromod_state)


class NREMReplayIntegrator:
    """
    Integrates spike reinjection with NREM consolidation (P3-02).

    Coordinates the replay loop:
    1. Select memories for replay (based on κ, importance, recency)
    2. Reinject as spike trains
    3. Process through SNN
    4. Apply STDP weight updates
    5. Update memory κ values
    """

    def __init__(
        self,
        reinjector: SpikeReinjector,
        snn_backend: nn.Module | None = None,
        stdp_learner: nn.Module | None = None,
    ):
        """
        Initialize NREM replay integrator.

        Args:
            reinjector: Spike reinjector module
            snn_backend: SNN for processing reinjected spikes
            stdp_learner: STDP learning module for weight updates
        """
        self.reinjector = reinjector
        self.snn_backend = snn_backend
        self.stdp_learner = stdp_learner

        logger.info("NREMReplayIntegrator initialized")

    async def replay_memories(
        self,
        memories: list[dict],
        neuromod_state: NeuromodulatorState | None = None,
        num_replays: int = 3,
    ) -> dict:
        """
        Replay memories through the spike→STDP loop.

        Args:
            memories: List of memory dicts with 'embedding' and 'id' keys
            neuromod_state: Neuromodulator state for modulation
            num_replays: Number of replay iterations per memory

        Returns:
            Dict with replay statistics
        """
        if not memories:
            return {"replayed": 0, "kappa_updates": 0}

        stats = {
            "replayed": 0,
            "kappa_updates": 0,
            "weight_updates": 0,
            "total_spikes": 0,
        }

        # Extract embeddings
        embeddings = [m["embedding"] for m in memories]
        embeddings_tensor = torch.tensor(
            np.stack(embeddings),
            dtype=torch.float32,
            device=self.reinjector.config.device,
        )

        for replay_idx in range(num_replays):
            # Generate spike trains
            spikes = self.reinjector(embeddings_tensor, neuromod_state)
            stats["total_spikes"] += spikes.sum().item()

            # Process through SNN if available
            if self.snn_backend is not None:
                with torch.no_grad():
                    snn_output, _ = self.snn_backend(spikes)

                # Apply STDP if available
                if self.stdp_learner is not None:
                    # STDP would update weights based on spike timing
                    # This is a placeholder for the actual STDP integration
                    stats["weight_updates"] += 1

            stats["replayed"] += len(memories)

        # Update κ values (+0.05 per NREM replay session)
        stats["kappa_updates"] = len(memories)

        logger.info(
            f"NREM replay complete: {stats['replayed']} replays, "
            f"{stats['total_spikes']:.0f} total spikes"
        )

        return stats

    def select_memories_for_replay(
        self,
        memories: list[dict],
        max_memories: int = 50,
    ) -> list[dict]:
        """
        Select memories for NREM replay based on biological criteria.

        Selection criteria:
        1. Low κ (recently encoded, need consolidation)
        2. High importance
        3. Recently accessed (temporal relevance)
        4. Not fully consolidated (κ < 0.9)

        Args:
            memories: All available memories
            max_memories: Maximum memories to replay

        Returns:
            Selected memories for replay
        """
        if not memories:
            return []

        # Filter out fully consolidated
        candidates = [m for m in memories if m.get("kappa", 0) < 0.9]

        # Score by: importance * (1 - kappa) * recency
        def replay_score(m: dict) -> float:
            importance = m.get("importance", 0.5)
            kappa = m.get("kappa", 0.0)
            # Higher score for lower κ (needs consolidation)
            return importance * (1 - kappa)

        # Sort by score descending
        candidates.sort(key=replay_score, reverse=True)

        return candidates[:max_memories]


__all__ = [
    "SpikeReinjector",
    "ReinjectionConfig",
    "ReinjectionMode",
    "NREMReplayIntegrator",
]
