"""
Dream-Based Negative Samples (W2-04).

Generate negative samples for Forward-Forward learning using VAE dreams.
More realistic negatives than random noise create tighter decision boundaries.

Evidence Base: Hinton (2022) "The Forward-Forward Algorithm"

Key Insight:
    Random noise negatives are too easy to distinguish from real data.
    VAE-generated "dreams" are closer to the data manifold but still
    different enough to serve as hard negatives, improving FF learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DreamNegativeConfig:
    """Configuration for dream-based negative generation.

    Attributes:
        corruption_rate: Fraction of output to corrupt with noise (default 0.3).
        default_method: Default generation method (default "vae_sample").
        latent_noise_scale: Scale for latent space noise (default 2.0).
    """

    corruption_rate: float = 0.3
    default_method: str = "vae_sample"
    latent_noise_scale: float = 2.0


class DreamNegativeGenerator:
    """Generate negative samples for FF learning using VAE dreams.

    Produces negative samples that are closer to the data manifold than
    random noise, creating harder negatives that improve FF learning.

    Methods:
    - vae_sample: Sample from VAE conditioned on positive, with corruption
    - shuffle: Shuffle features across batch dimension
    - hybrid: Mix VAE and shuffle methods

    Example:
        >>> generator = DreamNegativeGenerator(vae, corruption_rate=0.3)
        >>> positive = torch.randn(32, 1024)
        >>> negative = generator.generate(positive, method="vae_sample")
    """

    def __init__(
        self,
        vae: Any,
        corruption_rate: float = 0.3,
        latent_noise_scale: float = 2.0,
    ):
        """Initialize dream negative generator.

        Args:
            vae: VAE model with encode() and decode() methods.
            corruption_rate: Fraction of output to corrupt (0-1).
            latent_noise_scale: Scale for added latent noise.
        """
        self.vae = vae
        self.corruption_rate = corruption_rate
        self.latent_noise_scale = latent_noise_scale

    def generate(
        self,
        positive_data: torch.Tensor,
        method: str = "vae_sample",
    ) -> torch.Tensor:
        """Generate negative samples from positive data.

        Args:
            positive_data: [batch, features] positive samples.
            method: Generation method ("vae_sample", "shuffle", "hybrid").

        Returns:
            Negative samples with same shape as positive.

        Raises:
            ValueError: If method is unknown.
        """
        if method == "vae_sample":
            return self._vae_sample(positive_data)
        elif method == "shuffle":
            return self._shuffle(positive_data)
        elif method == "hybrid":
            return self._hybrid(positive_data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _vae_sample(self, positive_data: torch.Tensor) -> torch.Tensor:
        """Generate negatives via VAE sampling.

        Encodes positive to latent, samples with higher variance (more
        dreaming), decodes, and corrupts to ensure negativity.
        """
        # Encode positive to latent
        mu, logvar = self.vae.encode(positive_data)

        # Sample with higher variance (more dreaming)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std) * self.latent_noise_scale
        z = mu + std * noise

        # Decode
        negative = self.vae.decode(z)

        # Corrupt to ensure negativity
        mask = torch.rand_like(negative) < self.corruption_rate
        corruption = torch.randn_like(negative)
        negative = torch.where(mask, corruption, negative)

        return negative

    def _shuffle(self, positive_data: torch.Tensor) -> torch.Tensor:
        """Generate negatives by shuffling within batch.

        Simple but effective: permute samples so that each position
        gets a sample from a different batch item.
        """
        batch_size = positive_data.size(0)
        idx = torch.randperm(batch_size, device=positive_data.device)

        # Ensure permutation is actually different
        if batch_size > 1:
            # Keep shuffling until at least one position changes
            while torch.equal(idx, torch.arange(batch_size, device=positive_data.device)):
                idx = torch.randperm(batch_size, device=positive_data.device)

        return positive_data[idx]

    def _hybrid(self, positive_data: torch.Tensor) -> torch.Tensor:
        """Generate negatives mixing VAE and shuffle methods.

        50% of batch uses VAE sampling, 50% uses shuffle.
        """
        vae_neg = self._vae_sample(positive_data)
        shuffle_neg = self._shuffle(positive_data)

        # Random mask for mixing
        batch_size = positive_data.size(0)
        mask = torch.rand(batch_size, device=positive_data.device) < 0.5

        # Expand mask for broadcasting
        mask = mask.unsqueeze(-1).expand_as(positive_data)

        negative = torch.where(mask, vae_neg, shuffle_neg)

        return negative
