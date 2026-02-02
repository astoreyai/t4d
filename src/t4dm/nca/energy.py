"""
Energy-Based Learning for NCA Dynamics.

Theoretical Foundation (Hinton Perspective):
============================================

1. Modern Hopfield Networks (Ramsauer et al., 2020):
   - Classical: E = -0.5 * x^T W x (quadratic, limited capacity)
   - Modern: E = -beta^-1 * log(sum_i exp(beta * patterns_i @ query))
   - Exponential storage capacity (N ~ d instead of N ~ sqrt(d))
   - Retrieval via softmax attention over stored patterns

2. Contrastive Learning:
   - Positive phase: clamp visible units to data, let hidden settle
   - Negative phase: free-run entire system toward equilibrium
   - Update: delta_W = lr * (<v_i h_j>_data - <v_i h_j>_model)
   - Persistent CD maintains chains between updates for faster mixing

3. Forward-Forward Algorithm (Hinton, 2022):
   - Replace backprop with layer-local learning signals
   - Goodness function: sum of squared activities
   - Positive data: maximize goodness; negative data: minimize
   - No backward pass, no storing activations

4. Langevin Dynamics:
   - dx/dt = -grad_E(x) + sqrt(2*T) * noise
   - Temperature controls exploration vs. exploitation
   - Samples from exp(-E(x)/T) at equilibrium

Integration with T4DM:
==============================
- NT state U = [DA, 5HT, ACh, NE, GABA, Glu] as network activation
- Coupling matrix K from coupling.py as learned weights
- Cognitive states from attractors.py as energy minima
- Memory embeddings from Hopfield integration as stored patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.nca.attractors import StateTransitionManager
    from t4dm.nca.coupling import LearnableCoupling

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class LearningPhase(Enum):
    """Phase of contrastive learning."""

    POSITIVE = auto()  # Clamped to observed data
    NEGATIVE = auto()  # Free-running toward equilibrium
    INFERENCE = auto()  # Pattern retrieval (no learning)


@dataclass
class EnergyConfig:
    """Configuration for energy-based learning system."""

    # Temperature and dynamics
    temperature: float = 1.0  # Softmax temperature / Langevin noise scale
    temperature_min: float = 0.1  # Minimum for annealing
    temperature_decay: float = 0.995  # Per-step decay during annealing

    # Hopfield parameters
    hopfield_scale: float = 1.0  # Scale of coupling energy term
    hopfield_beta: float = 8.0  # Inverse temperature for modern Hopfield
    hopfield_beta_min: float = 2.0  # Minimum beta (high arousal = sharp retrieval)
    hopfield_beta_max: float = 16.0  # Maximum beta (low arousal = diffuse)
    arousal_modulates_beta: bool = True  # Enable NE arousal -> beta coupling

    # Single-unit potential (boundary enforcement)
    boundary_steepness: float = 10.0  # Sigmoid sharpness at bounds
    boundary_penalty: float = 5.0  # Energy penalty for out-of-bounds

    # Attractor landscape
    attractor_strength: float = 0.5  # Pull toward attractor centers
    metastability: float = 0.1  # Barrier height between attractors
    attractor_temperature: float = 0.5  # Softness of attractor wells

    # Contrastive learning
    learning_rate: float = 0.01
    contrastive_steps: int = 10  # CD-k steps for negative phase
    persistent_cd: bool = True  # Maintain chains between updates
    max_persistent_chains: int = 1000  # Max persistent chain steps before reset
    weight_decay: float = 0.0001  # L2 regularization

    # Langevin dynamics
    langevin_dt: float = 0.01  # Integration time step
    langevin_friction: float = 1.0  # Damping coefficient
    use_langevin: bool = True  # Use stochastic dynamics
    use_mh_correction: bool = True  # Metropolis-Hastings acceptance criterion

    # Forward-forward parameters
    ff_threshold: float = 2.0  # Goodness threshold for classification
    ff_learning_rate: float = 0.03  # Typically higher than CD

    # History tracking
    max_history: int = 10000


@dataclass
class ContrastiveState:
    """State maintained across contrastive divergence updates."""

    # Persistent chains for CD
    persistent_chain: np.ndarray | None = None
    chain_steps: int = 0

    # Statistics for monitoring convergence
    positive_energy_history: list = field(default_factory=list)
    negative_energy_history: list = field(default_factory=list)
    gradient_norm_history: list = field(default_factory=list)


# =============================================================================
# Energy Landscape
# =============================================================================


class EnergyLandscape:
    """
    Energy landscape for NT dynamics with learnable components.

    The total energy function is:
        E(U) = E_hopfield(U) + E_boundary(U) + E_attractor(U)

    where:
        E_hopfield = -0.5 * U^T K U (classic) or log-sum-exp (modern)
        E_boundary = soft penalty for U outside [0, 1]
        E_attractor = landscape with wells at cognitive states

    The system evolves via gradient descent (or Langevin dynamics):
        dU/dt = -grad_E(U) + sqrt(2T) * noise

    Learning occurs via contrastive divergence:
        delta_K = lr * (<U_i U_j>_data - <U_i U_j>_model)
    """

    def __init__(
        self,
        config: EnergyConfig | None = None,
        coupling: LearnableCoupling | None = None,
        state_manager: StateTransitionManager | None = None,
    ):
        """
        Initialize energy landscape.

        Args:
            config: Energy configuration parameters
            coupling: Learnable coupling matrix (weights)
            state_manager: Cognitive state manager for attractors
        """
        self.config = config or EnergyConfig()
        self.coupling = coupling
        self.state_manager = state_manager

        # Current temperature (can be annealed)
        self._temperature = self.config.temperature

        # Contrastive learning state
        self._cd_state = ContrastiveState()

        # Energy history for analysis
        self._energy_history: list[float] = []
        self._gradient_history: list[np.ndarray] = []

        # Random state for reproducibility
        self._rng = np.random.default_rng()

        # Arousal-modulated beta (Quick Win 1: NE -> Hopfield beta)
        self._current_beta = self.config.hopfield_beta

        logger.info(
            f"EnergyLandscape initialized: T={self._temperature:.2f}, "
            f"beta={self.config.hopfield_beta:.1f}"
        )

    # -------------------------------------------------------------------------
    # Arousal-Modulated Beta (Quick Win 1)
    # -------------------------------------------------------------------------

    def compute_arousal_beta(self, ne_level: float) -> float:
        """
        Compute Hopfield beta modulated by norepinephrine (NE) arousal.

        Biological basis (Hinton perspective):
        - High NE (arousal) -> sharper retrieval (higher beta)
        - Low NE (drowsy) -> more diffuse retrieval (lower beta)

        This implements attention-modulated memory retrieval:
        beta = beta_min + (beta_max - beta_min) * NE

        Args:
            ne_level: Norepinephrine level [0, 1]

        Returns:
            Arousal-modulated beta for Hopfield retrieval
        """
        if not self.config.arousal_modulates_beta:
            return self.config.hopfield_beta

        # Linear interpolation: high NE = high beta (sharp), low NE = low beta (diffuse)
        ne_clamped = np.clip(ne_level, 0.0, 1.0)
        beta = (
            self.config.hopfield_beta_min
            + (self.config.hopfield_beta_max - self.config.hopfield_beta_min) * ne_clamped
        )

        self._current_beta = float(beta)
        return self._current_beta

    def get_current_beta(self) -> float:
        """Get current arousal-modulated beta."""
        return self._current_beta

    # -------------------------------------------------------------------------
    # Energy Computation
    # -------------------------------------------------------------------------

    def compute_hopfield_energy(self, U: np.ndarray) -> float:
        """
        Compute Hopfield energy from coupling matrix.

        Classic formulation:
            E_hop = -0.5 * U^T K U

        This creates energy minima where strongly coupled NTs
        are simultaneously active or inactive.

        Args:
            U: NT state vector [6]

        Returns:
            Hopfield energy (scalar, lower = more stable)
        """
        if self.coupling is None:
            return 0.0

        K = self.coupling.K
        energy = -0.5 * self.config.hopfield_scale * float(U @ K @ U)
        return energy

    def compute_boundary_energy(self, U: np.ndarray) -> float:
        """
        Compute soft boundary potential enforcing [0, 1] range.

        Uses sigmoid-based soft barriers:
            V(u) = penalty * (softplus(-s*u) + softplus(s*(u-1)))

        This creates smooth energy wells that strongly penalize
        values outside the biological range without hard clipping.

        Args:
            U: NT state vector [6]

        Returns:
            Boundary penalty energy
        """
        s = self.config.boundary_steepness
        p = self.config.boundary_penalty

        # Softplus for smooth penalties: log(1 + exp(x))
        # Below 0: penalty grows as U decreases
        low_penalty = np.sum(np.log1p(np.exp(-s * U)))

        # Above 1: penalty grows as U increases
        high_penalty = np.sum(np.log1p(np.exp(s * (U - 1.0))))

        return float(p * (low_penalty + high_penalty) / s)

    def compute_attractor_energy(self, U: np.ndarray) -> float:
        """
        Compute energy from attractor landscape.

        Creates multi-well potential with:
        - Gaussian wells at each attractor center
        - Soft barriers between attractors (metastability)

        Uses log-sum-exp for smooth minimum:
            E_attr = -T * log(sum_i exp(-||U - c_i||^2 / (2*sigma_i^2) / T))

        Args:
            U: NT state vector [6]

        Returns:
            Attractor landscape energy
        """
        if self.state_manager is None:
            return 0.0

        T = self.config.attractor_temperature
        attractors = self.state_manager.attractors

        if not attractors:
            return 0.0

        # Compute negative log-sum-exp of Gaussian wells
        log_weights = []
        for state, attractor in attractors.items():
            dist_sq = float(np.sum((U - attractor.center) ** 2))
            width_sq = attractor.width**2

            # Gaussian well: stability * exp(-dist^2 / (2*width^2))
            log_weight = attractor.stability - dist_sq / (2 * width_sq * T)
            log_weights.append(log_weight)

        # Softmin via log-sum-exp trick for numerical stability
        max_log = max(log_weights)
        logsumexp = max_log + np.log(
            sum(np.exp(lw - max_log) for lw in log_weights)
        )

        # Energy is negative of this (wells are attractive)
        attractor_energy = -self.config.attractor_strength * T * logsumexp

        # Add metastability barrier (distance to nearest attractor)
        min_dist = min(
            float(np.linalg.norm(U - a.center)) for a in attractors.values()
        )
        barrier_energy = self.config.metastability * min_dist

        return float(attractor_energy + barrier_energy)

    def compute_total_energy(self, U: np.ndarray) -> float:
        """
        Compute total energy of NT configuration.

        E_total = E_hopfield + E_boundary + E_attractor

        Args:
            U: NT state vector [6]

        Returns:
            Total energy (lower = more stable)
        """
        e_hop = self.compute_hopfield_energy(U)
        e_bound = self.compute_boundary_energy(U)
        e_attr = self.compute_attractor_energy(U)

        total = e_hop + e_bound + e_attr

        # Track history
        self._energy_history.append(total)
        if len(self._energy_history) > self.config.max_history:
            self._energy_history = self._energy_history[-self.config.max_history :]

        return total

    # -------------------------------------------------------------------------
    # Analytical Gradients
    # -------------------------------------------------------------------------

    def compute_hopfield_gradient(self, U: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of Hopfield energy.

        grad_E_hop = -K @ U (symmetric K assumed)

        Args:
            U: NT state vector [6]

        Returns:
            Gradient vector [6]
        """
        if self.coupling is None:
            return np.zeros(6, dtype=np.float32)

        K = self.coupling.K
        # For symmetric K: grad(-0.5 * U^T K U) = -K @ U
        # For general K: grad = -0.5 * (K + K^T) @ U
        K_sym = 0.5 * (K + K.T)
        return -self.config.hopfield_scale * (K_sym @ U).astype(np.float32)

    def compute_boundary_gradient(self, U: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of boundary energy.

        grad_V(u) = penalty * (-sigmoid(-s*u) + sigmoid(s*(u-1)))

        Args:
            U: NT state vector [6]

        Returns:
            Gradient vector [6]
        """
        s = self.config.boundary_steepness
        p = self.config.boundary_penalty

        # Sigmoid function
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        # Gradient of softplus(x) is sigmoid(x)
        grad_low = -sigmoid(-s * U)  # Pushes up when U < 0
        grad_high = sigmoid(s * (U - 1.0))  # Pushes down when U > 1

        return (p * (grad_low + grad_high)).astype(np.float32)

    def compute_attractor_gradient(self, U: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of attractor energy.

        Uses softmax-weighted combination of gradients toward each attractor.

        Args:
            U: NT state vector [6]

        Returns:
            Gradient vector [6]
        """
        if self.state_manager is None:
            return np.zeros(6, dtype=np.float32)

        T = self.config.attractor_temperature
        attractors = self.state_manager.attractors

        if not attractors:
            return np.zeros(6, dtype=np.float32)

        # Compute softmax weights over attractors
        log_weights = []
        gradients = []

        for state, attractor in attractors.items():
            diff = U - attractor.center
            dist_sq = float(np.sum(diff**2))
            width_sq = attractor.width**2

            log_weight = attractor.stability - dist_sq / (2 * width_sq * T)
            log_weights.append(log_weight)

            # Gradient of Gaussian: -2*(U-c)/(2*width^2) = -(U-c)/width^2
            grad = diff / (width_sq * T)
            gradients.append(grad)

        # Softmax weights
        max_log = max(log_weights)
        exp_weights = np.array([np.exp(lw - max_log) for lw in log_weights])
        softmax_weights = exp_weights / np.sum(exp_weights)

        # Weighted gradient (points toward most relevant attractor)
        grad_attr = self.config.attractor_strength * sum(
            w * g for w, g in zip(softmax_weights, gradients)
        )

        # Metastability gradient (points away from nearest attractor)
        nearest_idx = np.argmin(
            [np.linalg.norm(U - a.center) for a in attractors.values()]
        )
        nearest = list(attractors.values())[nearest_idx]
        diff_nearest = U - nearest.center
        dist_nearest = np.linalg.norm(diff_nearest)
        if dist_nearest > 1e-6:
            grad_barrier = (
                self.config.metastability * diff_nearest / dist_nearest
            )
        else:
            grad_barrier = np.zeros(6)

        return (grad_attr + grad_barrier).astype(np.float32)

    def compute_energy_gradient(
        self, U: np.ndarray, numerical: bool = False, epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        Compute gradient of total energy.

        Args:
            U: NT state vector [6]
            numerical: Use numerical differentiation (for debugging)
            epsilon: Step size for numerical gradient

        Returns:
            Energy gradient [6]
        """
        if numerical:
            grad = np.zeros(6, dtype=np.float32)
            for i in range(6):
                U_plus = U.copy()
                U_plus[i] += epsilon
                U_minus = U.copy()
                U_minus[i] -= epsilon
                grad[i] = (
                    self.compute_total_energy(U_plus)
                    - self.compute_total_energy(U_minus)
                ) / (2 * epsilon)
            return grad

        # Analytical gradients
        grad = (
            self.compute_hopfield_gradient(U)
            + self.compute_boundary_gradient(U)
            + self.compute_attractor_gradient(U)
        )

        # Track gradient history
        self._gradient_history.append(grad.copy())
        if len(self._gradient_history) > self.config.max_history:
            self._gradient_history = self._gradient_history[
                -self.config.max_history :
            ]

        return grad

    # -------------------------------------------------------------------------
    # Dynamics
    # -------------------------------------------------------------------------

    def langevin_step(
        self,
        U: np.ndarray,
        dt: float | None = None,
        temperature: float | None = None,
    ) -> np.ndarray:
        """
        Langevin dynamics step for stochastic energy minimization.

        dU = -grad_E(U) * dt + sqrt(2*T*dt) * noise

        This samples from the Boltzmann distribution exp(-E(U)/T)
        at equilibrium, with T controlling exploration.

        Args:
            U: Current NT state [6]
            dt: Time step (default from config)
            temperature: Temperature (default current annealing temp)

        Returns:
            Updated NT state [6]
        """
        dt = dt or self.config.langevin_dt
        T = temperature if temperature is not None else self._temperature

        grad = self.compute_energy_gradient(U)

        # Deterministic drift
        drift = -grad * dt / self.config.langevin_friction

        # Stochastic diffusion
        if T > 0 and self.config.use_langevin:
            noise_scale = np.sqrt(2 * T * dt / self.config.langevin_friction)
            noise = self._rng.standard_normal(6).astype(np.float32) * noise_scale
        else:
            noise = 0

        U_new = U + drift + noise

        # ATOM-P3-38: Metropolis-Hastings correction
        if self.config.use_mh_correction and T > 0:
            E_old = self.compute_total_energy(U)
            E_new = self.compute_total_energy(U_new)
            delta_E = E_new - E_old
            accept_prob = min(1.0, np.exp(-delta_E / T))
            if self._rng.random() > accept_prob:
                U_new = U  # Reject

        # Soft clamping (let boundary energy handle it mostly)
        U_new = np.clip(U_new, -0.1, 1.1)

        return U_new.astype(np.float32)

    def gradient_step(
        self, U: np.ndarray, lr: float | None = None
    ) -> np.ndarray:
        """
        Simple gradient descent step (deterministic).

        Args:
            U: Current NT state [6]
            lr: Learning rate (default from config)

        Returns:
            Updated NT state [6]
        """
        lr = lr or self.config.learning_rate
        grad = self.compute_energy_gradient(U)
        U_new = U - lr * grad
        return np.clip(U_new, 0.0, 1.0).astype(np.float32)

    def relax_to_equilibrium(
        self,
        U: np.ndarray,
        max_steps: int = 100,
        tolerance: float = 1e-4,
        use_langevin: bool = True,
    ) -> tuple[np.ndarray, int, float]:
        """
        Relax NT state toward energy minimum.

        Args:
            U: Initial NT state [6]
            max_steps: Maximum relaxation steps
            tolerance: Convergence threshold for gradient norm
            use_langevin: Use stochastic dynamics

        Returns:
            (final_state, steps_taken, final_energy)
        """
        U_current = U.copy()
        prev_energy = self.compute_total_energy(U_current)

        for step in range(max_steps):
            if use_langevin:
                U_current = self.langevin_step(U_current)
            else:
                U_current = self.gradient_step(U_current)

            grad_norm = float(np.linalg.norm(self.compute_energy_gradient(U_current)))

            if grad_norm < tolerance:
                break

        final_energy = self.compute_total_energy(U_current)
        return U_current, step + 1, final_energy

    def anneal_temperature(self) -> float:
        """
        Decay temperature for simulated annealing.

        Returns:
            New temperature
        """
        self._temperature = max(
            self.config.temperature_min,
            self._temperature * self.config.temperature_decay,
        )
        return self._temperature

    def reset_temperature(self) -> None:
        """Reset temperature to initial value."""
        self._temperature = self.config.temperature

    # -------------------------------------------------------------------------
    # Contrastive Learning
    # -------------------------------------------------------------------------

    def contrastive_divergence_step(
        self,
        data_state: np.ndarray,
        k: int | None = None,
    ) -> dict:
        """
        Perform one contrastive divergence update on coupling matrix.

        Algorithm (CD-k):
        1. Positive phase: compute correlations at data point
        2. Negative phase: run k steps of dynamics, compute correlations
        3. Update: delta_K = lr * (positive_corr - negative_corr)

        With persistent CD, the negative chain continues from previous endpoint.

        Args:
            data_state: Observed NT state (clamped in positive phase) [6]
            k: Number of negative phase steps (default from config)

        Returns:
            Dictionary with learning statistics
        """
        if self.coupling is None:
            return {"error": "No coupling matrix to update"}

        k = k or self.config.contrastive_steps
        lr = self.config.learning_rate

        # ----- Positive Phase -----
        # Correlations at data point (clamped)
        U_pos = data_state.copy()
        positive_corr = np.outer(U_pos, U_pos)
        positive_energy = self.compute_total_energy(U_pos)

        # ----- Negative Phase -----
        # Start from persistent chain or data
        if self.config.persistent_cd and self._cd_state.persistent_chain is not None:
            U_neg = self._cd_state.persistent_chain.copy()
        else:
            U_neg = data_state.copy()

        # Run k steps of dynamics
        for _ in range(k):
            U_neg = self.langevin_step(U_neg)

        negative_corr = np.outer(U_neg, U_neg)
        negative_energy = self.compute_total_energy(U_neg)

        # Update persistent chain
        if self.config.persistent_cd:
            self._cd_state.persistent_chain = U_neg.copy()
            self._cd_state.chain_steps += k

            # Reset chain if it exceeds max steps to prevent unbounded growth
            if self._cd_state.chain_steps > self.config.max_persistent_chains:
                self._cd_state.persistent_chain = data_state.copy()
                self._cd_state.chain_steps = 0
                logger.debug(f"Reset persistent chain after {self._cd_state.chain_steps} steps")

        # ----- Weight Update -----
        gradient = positive_corr - negative_corr

        # Apply weight decay
        if self.config.weight_decay > 0:
            gradient -= self.config.weight_decay * self.coupling.K

        # Update coupling matrix
        self.coupling.K += lr * gradient

        # Enforce biological bounds
        if self.coupling.config.enforce_bounds:
            self.coupling.K = self.coupling.bounds.clamp(self.coupling.K)

        # Track statistics
        self._cd_state.positive_energy_history.append(positive_energy)
        self._cd_state.negative_energy_history.append(negative_energy)
        self._cd_state.gradient_norm_history.append(float(np.linalg.norm(gradient)))

        return {
            "positive_energy": positive_energy,
            "negative_energy": negative_energy,
            "energy_gap": positive_energy - negative_energy,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "k_steps": k,
            "persistent_chain_steps": self._cd_state.chain_steps,
        }

    def batch_contrastive_update(
        self,
        data_batch: list[np.ndarray],
        k: int | None = None,
    ) -> dict:
        """
        Contrastive divergence over a batch of observations.

        Args:
            data_batch: List of observed NT states
            k: Negative phase steps

        Returns:
            Aggregated learning statistics
        """
        if not data_batch:
            return {"error": "Empty batch"}

        stats_list = []
        for data_state in data_batch:
            stats = self.contrastive_divergence_step(data_state, k)
            stats_list.append(stats)

        return {
            "batch_size": len(data_batch),
            "mean_positive_energy": float(
                np.mean([s["positive_energy"] for s in stats_list])
            ),
            "mean_negative_energy": float(
                np.mean([s["negative_energy"] for s in stats_list])
            ),
            "mean_gradient_norm": float(
                np.mean([s["gradient_norm"] for s in stats_list])
            ),
        }

    # -------------------------------------------------------------------------
    # Forward-Forward Learning
    # -------------------------------------------------------------------------

    def compute_goodness(self, U: np.ndarray) -> float:
        """
        Compute forward-forward goodness function.

        Goodness = sum of squared activities (simpler than energy).

        High goodness indicates a "real" pattern, low indicates noise.

        Args:
            U: NT state vector [6]

        Returns:
            Goodness score (higher = more real)
        """
        return float(np.sum(U**2))

    def forward_forward_step(
        self,
        positive_data: np.ndarray,
        negative_data: np.ndarray,
    ) -> dict:
        """
        Forward-forward learning step (Hinton 2022).

        Instead of backprop, use layer-local learning:
        - Positive data: increase goodness
        - Negative data: decrease goodness

        For the NT system, we adjust coupling to:
        - Increase energy for negative samples (push away)
        - Decrease energy for positive samples (pull toward)

        Args:
            positive_data: Real/desired NT state [6]
            negative_data: Corrupted/negative NT state [6]

        Returns:
            Learning statistics
        """
        if self.coupling is None:
            return {"error": "No coupling matrix"}

        lr = self.config.ff_learning_rate
        threshold = self.config.ff_threshold

        # Compute goodness
        pos_goodness = self.compute_goodness(positive_data)
        neg_goodness = self.compute_goodness(negative_data)

        # Probability of "real" via sigmoid
        pos_prob = 1.0 / (1.0 + np.exp(threshold - pos_goodness))
        neg_prob = 1.0 / (1.0 + np.exp(threshold - neg_goodness))

        # Gradient for positive: increase goodness (decrease energy)
        # delta_K += lr * (1 - pos_prob) * outer(pos, pos)
        pos_grad = (1 - pos_prob) * np.outer(positive_data, positive_data)

        # Gradient for negative: decrease goodness (increase energy)
        # delta_K -= lr * neg_prob * outer(neg, neg)
        neg_grad = neg_prob * np.outer(negative_data, negative_data)

        # Combined update
        gradient = pos_grad - neg_grad
        self.coupling.K += lr * gradient

        # Enforce bounds
        if self.coupling.config.enforce_bounds:
            self.coupling.K = self.coupling.bounds.clamp(self.coupling.K)

        return {
            "positive_goodness": pos_goodness,
            "negative_goodness": neg_goodness,
            "positive_prob": pos_prob,
            "negative_prob": neg_prob,
            "gradient_norm": float(np.linalg.norm(gradient)),
        }

    def generate_negative_sample(
        self,
        positive_data: np.ndarray,
        method: str = "noise",
    ) -> np.ndarray:
        """
        Generate negative sample for forward-forward learning.

        Methods:
        - "noise": Add Gaussian noise
        - "shuffle": Random permutation of NT values
        - "adversarial": Gradient ascent on goodness

        Args:
            positive_data: Real NT state [6]
            method: Negative generation method

        Returns:
            Negative sample [6]
        """
        if method == "noise":
            noise = self._rng.standard_normal(6) * 0.3
            return np.clip(positive_data + noise, 0, 1).astype(np.float32)

        elif method == "shuffle":
            return self._rng.permutation(positive_data).astype(np.float32)

        elif method == "adversarial":
            # Gradient ascent on energy (toward high-energy states)
            neg = positive_data.copy()
            for _ in range(5):
                grad = self.compute_energy_gradient(neg)
                neg = neg + 0.1 * grad  # Ascent, not descent
                neg = np.clip(neg, 0, 1)
            return neg.astype(np.float32)

        else:
            raise ValueError(f"Unknown negative generation method: {method}")

    # -------------------------------------------------------------------------
    # Statistics and Persistence
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get comprehensive energy landscape statistics."""
        stats = {
            "temperature": self._temperature,
            "energy_history_size": len(self._energy_history),
            "config": {
                "temperature": self.config.temperature,
                "hopfield_scale": self.config.hopfield_scale,
                "hopfield_beta": self.config.hopfield_beta,
                "attractor_strength": self.config.attractor_strength,
                "contrastive_steps": self.config.contrastive_steps,
                "use_langevin": self.config.use_langevin,
            },
        }

        if self._energy_history:
            recent = self._energy_history[-100:]
            stats["recent_mean_energy"] = float(np.mean(recent))
            stats["recent_std_energy"] = float(np.std(recent))
            stats["recent_min_energy"] = float(np.min(recent))
            stats["recent_max_energy"] = float(np.max(recent))

        if self._gradient_history:
            recent_grad = self._gradient_history[-100:]
            norms = [float(np.linalg.norm(g)) for g in recent_grad]
            stats["recent_mean_gradient_norm"] = float(np.mean(norms))

        # CD statistics
        if self._cd_state.positive_energy_history:
            stats["cd_stats"] = {
                "updates": len(self._cd_state.positive_energy_history),
                "persistent_chain_steps": self._cd_state.chain_steps,
                "recent_energy_gap": (
                    float(
                        np.mean(self._cd_state.positive_energy_history[-10:])
                        - np.mean(self._cd_state.negative_energy_history[-10:])
                    )
                    if len(self._cd_state.positive_energy_history) >= 10
                    else None
                ),
            }

        return stats

    def save_state(self) -> dict:
        """Save energy landscape state for persistence."""
        state = {
            "temperature": self._temperature,
            "energy_history": self._energy_history[-1000:],  # Keep recent
            "cd_state": {
                "persistent_chain": (
                    self._cd_state.persistent_chain.tolist()
                    if self._cd_state.persistent_chain is not None
                    else None
                ),
                "chain_steps": self._cd_state.chain_steps,
            },
        }
        return state

    def load_state(self, state: dict) -> None:
        """Load energy landscape state from persistence."""
        self._temperature = state.get("temperature", self.config.temperature)
        self._energy_history = state.get("energy_history", [])

        cd = state.get("cd_state", {})
        if cd.get("persistent_chain") is not None:
            self._cd_state.persistent_chain = np.array(
                cd["persistent_chain"], dtype=np.float32
            )
        self._cd_state.chain_steps = cd.get("chain_steps", 0)


# =============================================================================
# Modern Hopfield Network for Memory Integration
# =============================================================================


class HopfieldIntegration:
    """
    Modern Hopfield network for memory pattern storage and retrieval.

    Implements the continuous Hopfield network (Ramsauer et al., 2020):
    - Energy: E = -beta^-1 * log(sum_i exp(beta * patterns_i @ query))
    - Retrieval: softmax attention over stored patterns
    - Capacity: Exponential in pattern dimension

    Each memory is associated with an NT state, enabling:
    - Context-dependent retrieval (current NT modulates retrieval)
    - Retrieval-dependent NT modulation (retrieved memory affects NT)
    """

    def __init__(
        self,
        dim: int = 768,
        num_patterns: int = 1000,
        beta: float = 8.0,
        separation_threshold: float = 0.1,
    ):
        """
        Initialize modern Hopfield integration.

        Args:
            dim: Dimension of memory embeddings
            num_patterns: Maximum stored patterns (soft limit)
            beta: Inverse temperature (higher = sharper retrieval)
            separation_threshold: Min distance between stored patterns
        """
        self.dim = dim
        self.num_patterns = num_patterns
        self.beta = beta
        self.separation_threshold = separation_threshold

        # Pattern storage
        self._patterns: list[np.ndarray] = []  # Memory embeddings
        self._pattern_states: list[np.ndarray] = []  # Associated NT states
        self._pattern_strengths: list[float] = []  # Retrieval strengths

        # Pattern norms for efficient computation
        self._pattern_norms: list[float] = []

        # Retrieval statistics
        self._retrieval_count = 0
        self._total_similarity = 0.0

        # Random state
        self._rng = np.random.default_rng()

        logger.info(
            f"HopfieldIntegration initialized: dim={dim}, "
            f"max_patterns={num_patterns}, beta={beta:.1f}"
        )

    # -------------------------------------------------------------------------
    # Pattern Storage
    # -------------------------------------------------------------------------

    def store_pattern(
        self,
        embedding: np.ndarray,
        nt_state: np.ndarray,
        strength: float = 1.0,
        consolidate: bool = True,
    ) -> int:
        """
        Store memory pattern with associated NT state.

        Args:
            embedding: Memory embedding [dim]
            nt_state: NT state during encoding [6]
            strength: Initial retrieval strength (can decay)
            consolidate: Merge similar patterns to prevent interference

        Returns:
            Pattern index (-1 if consolidated into existing)
        """
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Check for similar existing pattern
        if consolidate and self._patterns:
            similarities = self._compute_all_similarities(embedding)
            max_sim_idx = int(np.argmax(similarities))
            max_sim = similarities[max_sim_idx]

            if max_sim > 1.0 - self.separation_threshold:
                # Consolidate: update existing pattern via weighted average
                self._consolidate_pattern(max_sim_idx, embedding, nt_state, strength)
                return -1

        # Store new pattern
        if len(self._patterns) >= self.num_patterns:
            # Remove weakest pattern
            self._remove_weakest_pattern()

        self._patterns.append(embedding.copy())
        self._pattern_states.append(nt_state.copy())
        self._pattern_strengths.append(strength)
        self._pattern_norms.append(1.0)  # Already normalized

        return len(self._patterns) - 1

    def _consolidate_pattern(
        self,
        idx: int,
        new_embedding: np.ndarray,
        new_nt_state: np.ndarray,
        new_strength: float,
    ) -> None:
        """Consolidate new pattern into existing one."""
        # Weighted average based on strengths
        old_strength = self._pattern_strengths[idx]
        total_strength = old_strength + new_strength

        # Update embedding
        alpha = new_strength / total_strength
        combined = (1 - alpha) * self._patterns[idx] + alpha * new_embedding
        combined = combined / np.linalg.norm(combined)
        self._patterns[idx] = combined

        # Update NT state
        self._pattern_states[idx] = (
            (1 - alpha) * self._pattern_states[idx] + alpha * new_nt_state
        )

        # Increase strength (consolidation strengthens memory)
        self._pattern_strengths[idx] = min(total_strength, 5.0)

    def _remove_weakest_pattern(self) -> None:
        """Remove pattern with lowest strength."""
        if not self._patterns:
            return

        min_idx = int(np.argmin(self._pattern_strengths))
        del self._patterns[min_idx]
        del self._pattern_states[min_idx]
        del self._pattern_strengths[min_idx]
        del self._pattern_norms[min_idx]

    def _compute_all_similarities(self, query: np.ndarray) -> np.ndarray:
        """Compute similarities between query and all patterns."""
        if not self._patterns:
            return np.array([])

        patterns = np.stack(self._patterns)  # [N, dim]
        return patterns @ query  # Cosine similarity (normalized)

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def set_beta(self, beta: float) -> None:
        """
        Set Hopfield inverse temperature (beta).

        Quick Win 1: Enables arousal-modulated retrieval sharpness.

        Args:
            beta: New beta value (higher = sharper retrieval)
        """
        self.beta = float(np.clip(beta, 1.0, 32.0))

    def retrieve(
        self,
        query: np.ndarray,
        current_nt: np.ndarray | None = None,
        nt_modulation: float = 0.0,
        arousal_beta: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Retrieve pattern most similar to query.

        Uses modern Hopfield attention:
            attention = softmax(beta * patterns @ query)
            retrieved = attention @ patterns

        Optionally modulates by current NT state.

        Args:
            query: Query embedding [dim]
            current_nt: Current NT state for modulation [6]
            nt_modulation: How much NT state affects retrieval [0, 1]
            arousal_beta: Override beta with arousal-modulated value (Quick Win 1)

        Returns:
            (retrieved_embedding, associated_nt_state, max_similarity)
        """
        if not self._patterns:
            return query.copy(), (
                current_nt.copy() if current_nt is not None
                else np.full(6, 0.5, dtype=np.float32)
            ), 0.0

        # Quick Win 1: Use arousal-modulated beta if provided
        effective_beta = arousal_beta if arousal_beta is not None else self.beta

        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query_normalized = query / query_norm
        else:
            query_normalized = query

        # Compute similarities
        patterns = np.stack(self._patterns)  # [N, dim]
        similarities = patterns @ query_normalized  # [N]

        # NT modulation: patterns with similar NT states get bonus
        if current_nt is not None and nt_modulation > 0:
            nt_states = np.stack(self._pattern_states)  # [N, 6]
            nt_similarities = 1.0 - np.linalg.norm(
                nt_states - current_nt, axis=1
            ) / np.sqrt(6)
            similarities = (
                (1 - nt_modulation) * similarities
                + nt_modulation * nt_similarities
            )

        # Softmax attention (modern Hopfield)
        # Scale by sqrt(dim) for attention-like scaling
        scaled_sim = effective_beta * similarities / np.sqrt(self.dim)

        # ATOM-P4-5: Log-sum-exp trick for numerical stability
        max_sim = np.max(scaled_sim)
        scaled_sim_safe = scaled_sim - max_sim  # Subtract max before exp
        exp_sim = np.exp(scaled_sim_safe)
        attention = exp_sim / np.sum(exp_sim)

        # Weighted retrieval
        retrieved = np.sum(attention[:, None] * patterns, axis=0)
        retrieved_norm = np.linalg.norm(retrieved)
        if retrieved_norm > 0:
            retrieved = retrieved / retrieved_norm * query_norm

        # Associated NT state
        nt_states = np.stack(self._pattern_states)
        associated_nt = np.sum(attention[:, None] * nt_states, axis=0)

        # Statistics
        self._retrieval_count += 1
        self._total_similarity += float(np.max(similarities))

        # Update pattern strengths based on retrieval
        self._update_strengths_on_retrieval(attention)

        return retrieved, associated_nt.astype(np.float32), float(np.max(similarities))

    def _update_strengths_on_retrieval(self, attention: np.ndarray) -> None:
        """Update pattern strengths based on retrieval attention."""
        # Patterns that are retrieved get strengthened
        for i, att in enumerate(attention):
            if att > 0.1:  # Significant attention
                self._pattern_strengths[i] = min(
                    self._pattern_strengths[i] * 1.01, 5.0
                )
            else:
                # Slight decay for unretrieved patterns
                self._pattern_strengths[i] *= 0.9999

    # -------------------------------------------------------------------------
    # Energy Computation
    # -------------------------------------------------------------------------

    def compute_energy(self, query: np.ndarray) -> float:
        """
        Compute modern Hopfield energy for query.

        E = -beta^-1 * log(sum_i exp(beta * pattern_i @ query))

        Lower energy indicates query is closer to stored patterns.

        Args:
            query: Query embedding [dim]

        Returns:
            Hopfield energy
        """
        if not self._patterns:
            return 0.0

        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query_normalized = query / query_norm
        else:
            query_normalized = query

        patterns = np.stack(self._patterns)
        similarities = patterns @ query_normalized

        # Log-sum-exp with numerical stability
        scaled_sim = self.beta * similarities / np.sqrt(self.dim)
        max_sim = np.max(scaled_sim)
        logsumexp = max_sim + np.log(np.sum(np.exp(scaled_sim - max_sim)))

        return float(-logsumexp / self.beta)

    def compute_retrieval_gradient(self, query: np.ndarray) -> np.ndarray:
        """
        Compute gradient of Hopfield energy w.r.t. query.

        This points toward the retrieved pattern (energy minimum).

        Args:
            query: Query embedding [dim]

        Returns:
            Gradient [dim]
        """
        if not self._patterns:
            return np.zeros(self.dim, dtype=np.float32)

        # Normalize
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query_normalized = query / query_norm
        else:
            query_normalized = query

        patterns = np.stack(self._patterns)
        similarities = patterns @ query_normalized

        # Softmax attention
        scaled_sim = self.beta * similarities / np.sqrt(self.dim)
        max_sim = np.max(scaled_sim)
        exp_sim = np.exp(scaled_sim - max_sim)
        attention = exp_sim / np.sum(exp_sim)

        # Gradient: -sum(attention * pattern) / sqrt(dim)
        retrieved = np.sum(attention[:, None] * patterns, axis=0)
        gradient = -retrieved / np.sqrt(self.dim)

        return gradient.astype(np.float32)

    # -------------------------------------------------------------------------
    # Pattern Management
    # -------------------------------------------------------------------------

    def decay_strengths(self, factor: float = 0.99) -> None:
        """Apply uniform decay to all pattern strengths."""
        self._pattern_strengths = [s * factor for s in self._pattern_strengths]

    def prune_weak_patterns(self, threshold: float = 0.1) -> int:
        """Remove patterns below strength threshold."""
        initial_count = len(self._patterns)

        # Find indices to keep
        keep_indices = [
            i for i, s in enumerate(self._pattern_strengths) if s >= threshold
        ]

        # Rebuild lists
        self._patterns = [self._patterns[i] for i in keep_indices]
        self._pattern_states = [self._pattern_states[i] for i in keep_indices]
        self._pattern_strengths = [self._pattern_strengths[i] for i in keep_indices]
        self._pattern_norms = [self._pattern_norms[i] for i in keep_indices]

        return initial_count - len(self._patterns)

    def get_pattern_distribution(self) -> dict:
        """Analyze distribution of stored patterns."""
        if not self._patterns:
            return {"num_patterns": 0}

        strengths = np.array(self._pattern_strengths)
        patterns = np.stack(self._patterns)

        # Compute inter-pattern similarities
        sim_matrix = patterns @ patterns.T
        np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
        mean_sim = float(np.mean(sim_matrix))
        max_sim = float(np.max(sim_matrix))

        return {
            "num_patterns": len(self._patterns),
            "mean_strength": float(np.mean(strengths)),
            "std_strength": float(np.std(strengths)),
            "min_strength": float(np.min(strengths)),
            "max_strength": float(np.max(strengths)),
            "mean_inter_similarity": mean_sim,
            "max_inter_similarity": max_sim,
        }

    # -------------------------------------------------------------------------
    # Statistics and Persistence
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get Hopfield integration statistics."""
        stats = {
            "num_patterns": len(self._patterns),
            "dim": self.dim,
            "beta": self.beta,
            "max_patterns": self.num_patterns,
            "retrieval_count": self._retrieval_count,
            "mean_retrieval_similarity": (
                self._total_similarity / self._retrieval_count
                if self._retrieval_count > 0
                else 0.0
            ),
        }

        if self._patterns:
            stats["pattern_distribution"] = self.get_pattern_distribution()

        return stats

    def save_state(self) -> dict:
        """Save Hopfield state for persistence."""
        return {
            "patterns": [p.tolist() for p in self._patterns],
            "pattern_states": [s.tolist() for s in self._pattern_states],
            "pattern_strengths": self._pattern_strengths,
            "retrieval_count": self._retrieval_count,
            "total_similarity": self._total_similarity,
            "beta": self.beta,
        }

    def load_state(self, state: dict) -> None:
        """Load Hopfield state from persistence."""
        self._patterns = [np.array(p, dtype=np.float32) for p in state["patterns"]]
        self._pattern_states = [
            np.array(s, dtype=np.float32) for s in state["pattern_states"]
        ]
        self._pattern_strengths = state["pattern_strengths"]
        self._pattern_norms = [float(np.linalg.norm(p)) for p in self._patterns]
        self._retrieval_count = state.get("retrieval_count", 0)
        self._total_similarity = state.get("total_similarity", 0.0)
        if "beta" in state:
            self.beta = state["beta"]


# =============================================================================
# Integrated Energy-Based Learner
# =============================================================================


class EnergyBasedLearner:
    """
    Unified energy-based learning system integrating all components.

    Combines:
    - EnergyLandscape for NT dynamics
    - HopfieldIntegration for memory patterns
    - Contrastive and forward-forward learning
    """

    def __init__(
        self,
        energy_config: EnergyConfig | None = None,
        coupling: LearnableCoupling | None = None,
        state_manager: StateTransitionManager | None = None,
        hopfield_dim: int = 768,
        hopfield_capacity: int = 1000,
    ):
        """
        Initialize integrated energy-based learner.

        Args:
            energy_config: Configuration for energy landscape
            coupling: Learnable coupling matrix
            state_manager: Cognitive state manager
            hopfield_dim: Dimension for memory patterns
            hopfield_capacity: Maximum stored patterns
        """
        self.energy_landscape = EnergyLandscape(
            config=energy_config,
            coupling=coupling,
            state_manager=state_manager,
        )

        self.hopfield = HopfieldIntegration(
            dim=hopfield_dim,
            num_patterns=hopfield_capacity,
            beta=energy_config.hopfield_beta if energy_config else 8.0,
        )

        self._learning_steps = 0

        logger.info(
            f"EnergyBasedLearner initialized: "
            f"hopfield_dim={hopfield_dim}, capacity={hopfield_capacity}"
        )

    def learn_from_experience(
        self,
        nt_state: np.ndarray,
        memory_embedding: np.ndarray | None = None,
        reward: float = 0.0,
        method: str = "contrastive",
    ) -> dict:
        """
        Learn from a single experience.

        Args:
            nt_state: Observed NT state [6]
            memory_embedding: Associated memory (if any) [dim]
            reward: Reward signal for modulating learning
            method: "contrastive" or "forward_forward"

        Returns:
            Learning statistics
        """
        stats = {}

        # Learn NT dynamics via energy-based method
        if method == "contrastive":
            stats["energy"] = self.energy_landscape.contrastive_divergence_step(
                nt_state
            )
        elif method == "forward_forward":
            negative = self.energy_landscape.generate_negative_sample(nt_state)
            stats["energy"] = self.energy_landscape.forward_forward_step(
                nt_state, negative
            )

        # Store memory pattern if provided
        if memory_embedding is not None:
            pattern_idx = self.hopfield.store_pattern(
                memory_embedding,
                nt_state,
                strength=1.0 + reward,  # Reward boosts initial strength
            )
            stats["pattern_idx"] = pattern_idx

        self._learning_steps += 1
        stats["total_steps"] = self._learning_steps

        return stats

    def retrieve_with_context(
        self,
        query: np.ndarray,
        current_nt: np.ndarray,
        nt_modulation: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Retrieve memory with NT context modulation.

        Args:
            query: Query embedding [dim]
            current_nt: Current NT state [6]
            nt_modulation: How much NT affects retrieval [0, 1]

        Returns:
            (retrieved_embedding, suggested_nt_state, similarity)
        """
        return self.hopfield.retrieve(query, current_nt, nt_modulation)

    def get_stats(self) -> dict:
        """Get comprehensive learner statistics."""
        return {
            "learning_steps": self._learning_steps,
            "energy_landscape": self.energy_landscape.get_stats(),
            "hopfield": self.hopfield.get_stats(),
        }


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Configuration
    "EnergyConfig",
    "LearningPhase",
    "ContrastiveState",
    # Core classes
    "EnergyLandscape",
    "HopfieldIntegration",
    "EnergyBasedLearner",
]
