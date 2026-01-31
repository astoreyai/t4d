"""
Attractor network for associative memory and pattern completion.

Biological inspiration: Hippocampal CA3 autoassociative network (Hopfield, 1982)

The attractor network stores patterns as stable fixed points (attractors)
in a recurrent neural network. Partial or noisy cues settle into the
nearest stored pattern through energy minimization.

Features:
- Pattern storage via Hebbian learning (outer product rule)
- Content-addressable retrieval
- Noise tolerance and pattern completion
- Theoretical capacity ~0.14N patterns
"""

import uuid
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

# Security limits
MAX_DIM = 65536  # Maximum dimension to prevent memory exhaustion
MAX_SETTLING_STEPS = 1000  # Maximum settling iterations
MAX_BASIN_SAMPLES = 1000  # Maximum samples for basin estimation


@dataclass
class AttractorConfig:
    """Configuration for attractor network."""
    dim: int = 8192              # Pattern dimension (matches sparse encoder)
    symmetric: bool = True       # Enforce symmetric weights
    noise_std: float = 0.01      # Internal noise for exploration
    settling_steps: int = 10     # Default settling iterations
    step_size: float = 0.1       # Update step size
    capacity_ratio: float = 0.138  # Hopfield capacity limit


@dataclass
class RetrievalResult:
    """Result from attractor network retrieval."""
    pattern: torch.Tensor
    pattern_id: str | None
    confidence: float
    steps: int
    energy: float
    trajectory: list[dict[str, Any]] | None = None


class AttractorNetwork:
    """
    Modern Hopfield network for associative memory.

    Stores sparse patterns as attractors with ~0.14N capacity.
    Retrieval via energy minimization (settling dynamics).

    The energy function is:
        E = -0.5 * s^T W s

    Update rule:
        s(t+1) = s(t) + α * (W @ s(t) - s(t))
    """

    def __init__(
        self,
        dim: int = 8192,
        symmetric: bool = True,
        noise_std: float = 0.01,
        settling_steps: int = 10,
        step_size: float = 0.1,
        capacity_ratio: float = 0.138,
        device: str | None = None
    ):
        # Security validation
        if dim > MAX_DIM:
            raise ValueError(f"dim ({dim}) exceeds MAX_DIM ({MAX_DIM})")
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        if settling_steps > MAX_SETTLING_STEPS:
            raise ValueError(f"settling_steps ({settling_steps}) exceeds MAX_SETTLING_STEPS ({MAX_SETTLING_STEPS})")
        if settling_steps < 1:
            raise ValueError(f"settling_steps must be positive, got {settling_steps}")
        if not 0 < capacity_ratio <= 1:
            raise ValueError(f"capacity_ratio must be in (0, 1], got {capacity_ratio}")

        self.dim = dim
        self.symmetric = symmetric
        self.noise_std = noise_std
        self.settling_steps = settling_steps
        self.step_size = step_size
        self.capacity = max(1, int(dim * capacity_ratio))  # Ensure at least 1

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Weight matrix (stored patterns create attractors)
        self.W = torch.zeros(dim, dim, device=device)

        # Pattern storage
        self.patterns: dict[str, torch.Tensor] = {}

    def store(
        self,
        pattern: torch.Tensor,
        pattern_id: str | None = None
    ) -> dict[str, Any]:
        """
        Store pattern as attractor via Hebbian learning.

        Uses outer product rule: ΔW = pattern ⊗ pattern

        Args:
            pattern: Pattern to store (dim,)
            pattern_id: Unique identifier (auto-generated if None)

        Returns:
            Dict with storage info including overlap with existing patterns
        """
        # Generate ID if needed
        if pattern_id is None:
            pattern_id = str(uuid.uuid4())[:8]

        # Normalize pattern
        pattern = pattern.to(self.device)
        pattern_norm = pattern / (pattern.norm() + 1e-8)

        # Check capacity
        if len(self.patterns) >= self.capacity:
            return {
                "stored": False,
                "pattern_id": pattern_id,
                "reason": f"Capacity reached ({self.capacity})",
                "capacity_usage": 1.0
            }

        # Compute overlap with existing patterns
        max_overlap = 0.0
        for stored_pattern in self.patterns.values():
            overlap = F.cosine_similarity(
                pattern_norm.unsqueeze(0),
                stored_pattern.unsqueeze(0)
            ).item()
            max_overlap = max(max_overlap, overlap)

        # Hebbian update: ΔW = pattern ⊗ pattern
        update = torch.outer(pattern_norm, pattern_norm)

        # Remove diagonal (no self-connections)
        update.fill_diagonal_(0)

        # Update weights
        self.W += update

        # Enforce symmetry if required
        if self.symmetric:
            self.W = (self.W + self.W.t()) / 2

        # Store pattern
        self.patterns[pattern_id] = pattern_norm

        return {
            "stored": True,
            "pattern_id": pattern_id,
            "max_overlap_with_existing": max_overlap,
            "capacity_usage": len(self.patterns) / self.capacity
        }

    def retrieve(
        self,
        cue: torch.Tensor,
        max_steps: int | None = None,
        track_trajectory: bool = False,
        convergence_threshold: float = 1e-6
    ) -> RetrievalResult:
        """
        Retrieve pattern via settling dynamics.

        Dynamics: s(t+1) = s(t) + α * (W @ s(t) - s(t))

        Args:
            cue: Initial state / query pattern
            max_steps: Maximum settling iterations
            track_trajectory: Store state at each step
            convergence_threshold: Energy change threshold for early stopping

        Returns:
            RetrievalResult with settled pattern and metadata
        """
        max_steps = max_steps or self.settling_steps
        max_steps = min(max_steps, MAX_SETTLING_STEPS)  # Security cap
        cue = cue.to(self.device)

        # Initialize state
        state = cue.clone()

        energies = []
        trajectory = [] if track_trajectory else None
        converged_step = max_steps

        for step in range(max_steps):
            # Compute energy: E = -0.5 * s^T W s
            energy = self.compute_energy(state)
            energies.append(energy)

            if track_trajectory:
                trajectory.append({
                    "step": step,
                    "state": state.clone().cpu(),
                    "energy": energy
                })

            # Check convergence
            if len(energies) > 1:
                energy_change = abs(energies[-1] - energies[-2])
                if energy_change < convergence_threshold:
                    converged_step = step
                    break

            # Update rule: s(t+1) = s(t) + α * (W @ s(t) - s(t))
            activation = self.W @ state
            state = state + self.step_size * (activation - state)

            # Optional: add small noise for exploration
            if self.noise_std > 0 and step < max_steps - 1:
                state = state + self.noise_std * torch.randn_like(state)

        # Find nearest stored pattern
        best_match = None
        best_similarity = -1.0

        for pid, pattern in self.patterns.items():
            sim = F.cosine_similarity(
                state.unsqueeze(0),
                pattern.unsqueeze(0)
            ).item()
            if sim > best_similarity:
                best_similarity = sim
                best_match = pid

        return RetrievalResult(
            pattern=state,
            pattern_id=best_match,
            confidence=best_similarity,
            steps=converged_step,
            energy=energies[-1] if energies else 0.0,
            trajectory=trajectory
        )

    def compute_energy(self, state: torch.Tensor) -> float:
        """
        Compute Hopfield energy.

        E = -0.5 * s^T W s

        Lower energy = more stable state.
        """
        state = state.to(self.device)
        return (-0.5 * torch.dot(state, self.W @ state)).item()

    def remove(self, pattern_id: str) -> bool:
        """
        Remove pattern from network (unlearn).

        Uses negative Hebbian update.

        Args:
            pattern_id: ID of pattern to remove

        Returns:
            True if pattern was removed
        """
        if pattern_id not in self.patterns:
            return False

        pattern = self.patterns[pattern_id]

        # Negative Hebbian update
        update = torch.outer(pattern, pattern)
        update.fill_diagonal_(0)
        self.W -= update

        if self.symmetric:
            self.W = (self.W + self.W.t()) / 2

        del self.patterns[pattern_id]
        return True

    def get_basin_estimate(
        self,
        pattern_id: str,
        num_samples: int = 100,
        noise_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
    ) -> dict[str, float]:
        """
        Estimate basin of attraction size for a pattern.

        Tests retrieval accuracy at various noise levels.

        Args:
            pattern_id: Pattern to test
            num_samples: Samples per noise level
            noise_levels: List of noise standard deviations

        Returns:
            Dict mapping noise level to retrieval accuracy
        """
        if pattern_id not in self.patterns:
            return {}

        # Security validation
        num_samples = min(num_samples, MAX_BASIN_SAMPLES)

        pattern = self.patterns[pattern_id]
        results = {}

        for noise in noise_levels:
            correct = 0
            for _ in range(num_samples):
                noisy_cue = pattern + noise * torch.randn_like(pattern)
                result = self.retrieve(noisy_cue)
                if result.pattern_id == pattern_id:
                    correct += 1
            results[noise] = correct / num_samples

        return results

    def analyze(self) -> dict[str, Any]:
        """
        Analyze current state of attractor network.

        Returns:
            Dict with capacity usage, weight statistics, etc.
        """
        # Compute pattern overlaps
        overlaps = []
        pattern_list = list(self.patterns.values())

        for i, p1 in enumerate(pattern_list):
            for p2 in pattern_list[i + 1:]:
                overlap = F.cosine_similarity(
                    p1.unsqueeze(0), p2.unsqueeze(0)
                ).item()
                overlaps.append(abs(overlap))

        return {
            "pattern_count": len(self.patterns),
            "theoretical_capacity": self.capacity,
            "capacity_usage": len(self.patterns) / self.capacity,
            "weight_matrix_norm": self.W.norm().item(),
            "weight_matrix_sparsity": (self.W == 0).float().mean().item(),
            "average_pattern_overlap": sum(overlaps) / len(overlaps) if overlaps else 0,
            "max_pattern_overlap": max(overlaps) if overlaps else 0,
        }

    def clear(self):
        """Clear all stored patterns and reset weights."""
        self.W.zero_()
        self.patterns.clear()

    @property
    def pattern_count(self) -> int:
        """Number of stored patterns."""
        return len(self.patterns)

    @property
    def usage_ratio(self) -> float:
        """Capacity usage ratio."""
        return len(self.patterns) / self.capacity


class ModernHopfieldNetwork(AttractorNetwork):
    """
    Modern Hopfield Network with exponential energy function.

    Uses softmax-based update for increased capacity.
    Capacity scales as O(d^α) where α > 1.

    Reference: Ramsauer et al. (2020) "Hopfield Networks is All You Need"

    Quick Win 1: Supports arousal-modulated beta (NE -> retrieval sharpness)
    """

    def __init__(
        self,
        dim: int = 8192,
        beta: float = 1.0,  # Inverse temperature
        beta_min: float = 0.5,  # Minimum beta (diffuse retrieval)
        beta_max: float = 4.0,  # Maximum beta (sharp retrieval)
        **kwargs
    ):
        super().__init__(dim=dim, **kwargs)
        self.beta = beta
        self.beta_min = beta_min
        self.beta_max = beta_max

    def set_beta_from_arousal(self, ne_level: float) -> float:
        """
        Set beta based on norepinephrine (NE) arousal level.

        Quick Win 1: High arousal (NE) -> sharper retrieval (higher beta)

        Args:
            ne_level: Norepinephrine level [0, 1]

        Returns:
            New beta value
        """
        ne_clamped = max(0.0, min(1.0, ne_level))
        self.beta = self.beta_min + (self.beta_max - self.beta_min) * ne_clamped
        return self.beta

    def retrieve(
        self,
        cue: torch.Tensor,
        max_steps: int | None = None,
        arousal_beta: float | None = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve using modern Hopfield dynamics.

        Uses attention-like softmax update.

        Args:
            cue: Query pattern for retrieval
            max_steps: Maximum settling iterations
            arousal_beta: Override beta with arousal-modulated value (Quick Win 1)
        """
        max_steps = max_steps or self.settling_steps
        max_steps = min(max_steps, MAX_SETTLING_STEPS)  # Security cap
        cue = cue.to(self.device)
        state = cue.clone()

        # Quick Win 1: Use arousal-modulated beta if provided
        effective_beta = arousal_beta if arousal_beta is not None else self.beta

        if not self.patterns:
            return RetrievalResult(
                pattern=state,
                pattern_id=None,
                confidence=0.0,
                steps=0,
                energy=0.0
            )

        # Stack patterns for vectorized computation
        patterns_tensor = torch.stack(list(self.patterns.values()))
        pattern_ids = list(self.patterns.keys())

        energies = []

        for step in range(max_steps):
            # Compute similarities (like attention scores)
            similarities = torch.mv(patterns_tensor, state) * effective_beta

            # Softmax weights
            weights = F.softmax(similarities, dim=0)

            # Energy (log-sum-exp)
            energy = -torch.logsumexp(similarities, dim=0).item() / self.beta
            energies.append(energy)

            # Update: weighted sum of patterns
            new_state = torch.mv(patterns_tensor.t(), weights)

            # Check convergence
            if torch.norm(new_state - state) < 1e-6:
                break

            state = new_state

        # Find best match
        final_similarities = torch.mv(patterns_tensor, state)
        best_idx = final_similarities.argmax().item()

        return RetrievalResult(
            pattern=state,
            pattern_id=pattern_ids[best_idx],
            confidence=F.softmax(final_similarities * effective_beta, dim=0)[best_idx].item(),
            steps=step + 1,
            energy=energies[-1] if energies else 0.0
        )
