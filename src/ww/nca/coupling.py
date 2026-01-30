"""
Learnable Coupling Matrix for NT Interactions.

Biological Basis (CompBio):
- NT interactions are NOT independent
- DA-NE antagonism, GABA-Glu balance, ACh-DA striatal interaction
- Coupling should be learnable but biologically bounded

Key Innovation (Hinton):
- Make KATIE's static K matrix learnable: K = nn.Parameter(init_K)
- Constrain to biological bounds via clamping
- Learn from prediction errors using energy-based methods

Coupling Equation:
    C_i = Σ_j K_ij × f(U_j)

where:
    K_ij = coupling coefficient (source j -> target i)
    f() = activation function (sigmoid for bounded NT)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BiologicalBounds:
    """
    Coupling coefficient bounds from CompBio analysis.

    Matrix layout: K[source, target]
    Order: DA, 5HT, ACh, NE, GABA, Glu

    Positive = excitatory interaction
    Negative = inhibitory interaction
    """

    # Minimum coupling values (biological floor)
    K_MIN: np.ndarray = None

    # Maximum coupling values (biological ceiling)
    K_MAX: np.ndarray = None

    def __post_init__(self):
        if self.K_MIN is None:
            # Default bounds based on CompBio neuroscience analysis
            # [source affects target]
            self.K_MIN = np.array([
                # DA    5HT   ACh   NE    GABA  Glu   <- target
                [-0.1, -0.1, -0.2, -0.4, -0.2,  0.0],  # DA source
                [-0.2,  0.0, -0.1, -0.1,  0.0, -0.1],  # 5HT source
                [-0.1, -0.1,  0.0, -0.1, -0.1,  0.0],  # ACh source
                [-0.1, -0.1,  0.0,  0.0, -0.3,  0.0],  # NE source
                [-0.2,  0.0, -0.1,  0.0,  0.0, -0.3],  # GABA source
                [ 0.0, -0.1,  0.0,  0.0, -0.3,  0.0],  # Glu source
            ], dtype=np.float32)

        if self.K_MAX is None:
            self.K_MAX = np.array([
                # DA    5HT   ACh   NE    GABA  Glu   <- target
                [ 0.0,  0.2,  0.1,  0.0,  0.1,  0.1],  # DA source
                [ 0.0,  0.0,  0.1,  0.1,  0.1,  0.1],  # 5HT source
                [ 0.2,  0.1,  0.0,  0.2,  0.1,  0.2],  # ACh source
                [ 0.2,  0.1,  0.2,  0.0,  0.0,  0.2],  # NE source
                [ 0.0,  0.1,  0.1,  0.2,  0.0,  0.0],  # GABA source
                [ 0.1,  0.1,  0.2,  0.1,  0.0,  0.0],  # Glu source
            ], dtype=np.float32)

    def clamp(self, K: np.ndarray) -> np.ndarray:
        """Clamp coupling matrix to biological bounds."""
        return np.clip(K, self.K_MIN, self.K_MAX)

    def is_valid(self, K: np.ndarray) -> bool:
        """Check if coupling matrix is within bounds."""
        return np.all(K >= self.K_MIN) and np.all(K <= self.K_MAX)


@dataclass
class CouplingConfig:
    """Configuration for learnable coupling."""

    # Initial coupling (KATIE defaults)
    init_coupling: np.ndarray | None = None

    # Learning parameters
    learning_rate: float = 0.01
    regularization: float = 0.001  # Prefer sparse coupling

    # Energy-based learning (Hinton)
    use_energy_learning: bool = True
    contrastive_steps: int = 5

    # Biological constraints
    enforce_bounds: bool = True
    enforce_ei_balance: bool = True  # GABA-Glu balance

    def __post_init__(self):
        if self.init_coupling is None:
            # Initialize at biological midpoint
            bounds = BiologicalBounds()
            self.init_coupling = (bounds.K_MIN + bounds.K_MAX) / 2


class LearnableCoupling:
    """
    Learnable coupling matrix with biological constraints.

    This is the KEY INNOVATION: making KATIE's static K matrix
    learnable via gradient-based or energy-based methods.

    Learning Methods:
    - update_from_rpe: Three-factor Hebbian (eligibility x dopamine x correlation)
    - update_from_energy: Contrastive divergence (Boltzmann machine style)
    - update_with_eligibility: Convenience method using accumulated traces
    """

    def __init__(
        self,
        config: CouplingConfig | None = None,
        bounds: BiologicalBounds | None = None
    ):
        """
        Initialize learnable coupling.

        Args:
            config: Learning configuration
            bounds: Biological plausibility bounds
        """
        self.config = config or CouplingConfig()
        self.bounds = bounds or BiologicalBounds()

        # Initialize coupling matrix
        self.K = self.config.init_coupling.copy()

        # Ensure within bounds
        if self.config.enforce_bounds:
            self.K = self.bounds.clamp(self.K)

        # Track learning history
        self._update_count = 0
        self._gradient_history: list[np.ndarray] = []

        # Quick Win 2: Eligibility trace accumulator
        self._eligibility_trace: np.ndarray = np.zeros((6, 6), dtype=np.float32)
        self._eligibility_decay: float = 0.9  # Decay rate per step
        self._nt_state_history: list[np.ndarray] = []
        self._max_history: int = 100

        # ATOM-P2-10: Rate limiting for learning updates
        self._last_update_time: float = 0.0
        self._update_count_window: int = 0
        self._max_updates_per_second: int = 100

        # ATOM-P2-16: Learning drift/anomaly detection
        self._weight_history: list[float] = []  # Track weight norm over time
        self._drift_window: int = 100

        logger.info(
            f"LearnableCoupling initialized: "
            f"bounds_valid={self.bounds.is_valid(self.K)}"
        )

    def compute_coupling(
        self,
        nt_state: NeurotransmitterState
    ) -> np.ndarray:
        """
        Compute coupling contribution to NT dynamics.

        C_i = Σ_j K_ij × sigmoid(U_j)

        Args:
            nt_state: Current NT concentrations

        Returns:
            Coupling contribution for each NT [6,]
        """
        from ww.nca.neural_field import NeurotransmitterState

        if isinstance(nt_state, NeurotransmitterState):
            U = nt_state.to_array()
        else:
            U = np.asarray(nt_state)

        # Sigmoid activation for bounded interactions
        activated = 1 / (1 + np.exp(-5 * (U - 0.5)))  # Centered sigmoid

        # Matrix multiply: K @ activated gives coupling for each target
        coupling = self.K.T @ activated  # [6,] output

        return coupling

    # -------------------------------------------------------------------------
    # Quick Win 2: Eligibility Trace Management
    # -------------------------------------------------------------------------

    def accumulate_eligibility(self, nt_state: NeurotransmitterState) -> None:
        """
        Accumulate eligibility trace from current NT state.

        Three-factor learning (Hinton): Eligibility traces mark synapses
        that were recently active. When dopamine arrives, these traces
        determine which synapses get updated.

        E(t) = decay * E(t-1) + outer(activated, activated)

        Args:
            nt_state: Current NT state to add to eligibility
        """
        from ww.nca.neural_field import NeurotransmitterState

        if isinstance(nt_state, NeurotransmitterState):
            U = nt_state.to_array()
        else:
            U = np.asarray(nt_state)

        # Sigmoid activation (same as compute_coupling)
        activated = 1 / (1 + np.exp(-5 * (U - 0.5)))

        # Decay existing trace
        self._eligibility_trace *= self._eligibility_decay

        # Add new contribution
        self._eligibility_trace += np.outer(activated, activated)

        # Cap eligibility trace magnitude to prevent unbounded growth
        self._eligibility_trace = np.clip(self._eligibility_trace, -10.0, 10.0)

        # Track state history
        self._nt_state_history.append(U.copy())
        if len(self._nt_state_history) > self._max_history:
            self._nt_state_history = self._nt_state_history[-self._max_history:]

    def get_eligibility_trace(self) -> np.ndarray:
        """Get current eligibility trace matrix [6, 6]."""
        return self._eligibility_trace.copy()

    def reset_eligibility(self) -> None:
        """Reset eligibility trace to zero."""
        self._eligibility_trace = np.zeros((6, 6), dtype=np.float32)
        self._nt_state_history = []

    def set_eligibility_decay(self, decay: float) -> None:
        """
        Set eligibility trace decay rate.

        Args:
            decay: Decay rate per step [0.8, 0.99]
        """
        self._eligibility_decay = float(np.clip(decay, 0.8, 0.99))

    def update_with_eligibility(
        self,
        nt_state: NeurotransmitterState,
        rpe: float,
        use_accumulated: bool = True
    ) -> None:
        """
        Update coupling using accumulated eligibility traces.

        Quick Win 2: Convenience method that uses the internal trace.

        Args:
            nt_state: Current NT state
            rpe: Reward prediction error (dopamine signal)
            use_accumulated: Use accumulated trace (True) or fresh (False)
        """
        eligibility = self._eligibility_trace if use_accumulated else None
        self.update_from_rpe(nt_state, rpe, eligibility)

        # Reset eligibility after use (RPE consumed the trace)
        if use_accumulated:
            self._eligibility_trace *= 0.5  # Partial reset

    def update_from_rpe(
        self,
        nt_state: NeurotransmitterState,
        rpe: float,
        eligibility: np.ndarray | None = None
    ) -> None:
        """
        Update coupling based on reward prediction error.

        Three-factor update: dK = lr × E × δ × ∂C/∂K

        Args:
            nt_state: NT state during retrieval
            rpe: Dopamine reward prediction error
            eligibility: Eligibility trace (optional)
        """
        import time
        from ww.nca.neural_field import NeurotransmitterState

        # ATOM-P2-10: Rate limiting for learning updates
        now = time.monotonic()
        if now - self._last_update_time < 1.0:
            self._update_count_window += 1
            if self._update_count_window > self._max_updates_per_second:
                raise ValueError("Learning rate limit exceeded: >100 updates/second")
        else:
            self._last_update_time = now
            self._update_count_window = 1

        if isinstance(nt_state, NeurotransmitterState):
            U = nt_state.to_array()
        else:
            U = np.asarray(nt_state)

        # Sigmoid activation
        activated = 1 / (1 + np.exp(-5 * (U - 0.5)))

        # FIXED: Proper gradient using outer product of activations
        # grad[i,j] = activated[i] * activated[j] represents Hebbian correlation
        # This respects specific source-target relationships
        grad = np.outer(activated, activated)  # [6, 6] Hebbian-style

        # Modulate by RPE (surprise drives learning)
        grad = grad * rpe

        # Apply eligibility if provided
        if eligibility is not None:
            grad = grad * eligibility

        # Update with learning rate
        self.K += self.config.learning_rate * grad

        # Regularization (prefer sparse)
        self.K -= self.config.regularization * self.K

        # Enforce biological bounds
        if self.config.enforce_bounds:
            self.K = self.bounds.clamp(self.K)

        # E/I balance constraint
        if self.config.enforce_ei_balance:
            self._enforce_ei_balance()

        self._update_count += 1
        self._gradient_history.append(grad.copy())

        # Limit history size
        if len(self._gradient_history) > 1000:
            self._gradient_history = self._gradient_history[-500:]

        # ATOM-P2-16: Learning drift/anomaly detection
        weight_norm = np.linalg.norm(self.K)
        self._weight_history.append(weight_norm)
        if len(self._weight_history) > self._drift_window:
            self._weight_history = self._weight_history[-self._drift_window:]
            mean_norm = np.mean(self._weight_history)
            std_norm = np.std(self._weight_history)
            if std_norm > 0 and abs(weight_norm - mean_norm) > 3 * std_norm:
                logger.warning(
                    f"Learning drift detected: weight norm {weight_norm:.4f} is "
                    f"{abs(weight_norm - mean_norm)/std_norm:.1f} sigma from mean {mean_norm:.4f}"
                )

    def update_from_energy(
        self,
        data_state: np.ndarray,
        model_state: np.ndarray | None = None,
        n_gibbs_steps: int = 5,
        eligibility: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Update coupling using contrastive divergence from energy function.

        Biological Basis (Hinton):
        - The coupling matrix defines an energy landscape
        - States with lower energy are more probable
        - Learning minimizes energy of observed states, maximizes model samples

        Contrastive Divergence:
            grad = <s_i * s_j>_data - <s_i * s_j>_model

        The positive phase uses clamped data, negative phase uses Gibbs samples.

        Args:
            data_state: Observed NT state (clamped, positive phase)
            model_state: Model equilibrium state (if None, run Gibbs sampling)
            n_gibbs_steps: Number of Gibbs steps for negative phase
            eligibility: Optional eligibility trace for three-factor learning

        Returns:
            Computed gradient for analysis
        """
        data_state = np.asarray(data_state).flatten()

        # Positive phase: correlation under data
        data_activated = 1 / (1 + np.exp(-5 * (data_state - 0.5)))
        pos_corr = np.outer(data_activated, data_activated)

        # Negative phase: correlation under model
        if model_state is None:
            # Run Gibbs sampling to get model samples
            model_state = self._gibbs_sample(data_state, n_gibbs_steps)

        model_activated = 1 / (1 + np.exp(-5 * (model_state - 0.5)))
        neg_corr = np.outer(model_activated, model_activated)

        # Contrastive divergence gradient
        grad = pos_corr - neg_corr

        # Apply eligibility if provided (three-factor learning)
        if eligibility is not None:
            grad = grad * eligibility

        # Update coupling
        self.K += self.config.learning_rate * grad

        # Regularization (L2)
        self.K -= self.config.regularization * self.K

        # Enforce constraints
        if self.config.enforce_bounds:
            self.K = self.bounds.clamp(self.K)
        if self.config.enforce_ei_balance:
            self._enforce_ei_balance()

        self._update_count += 1
        self._gradient_history.append(grad.copy())

        if len(self._gradient_history) > 1000:
            self._gradient_history = self._gradient_history[-500:]

        return grad

    def _gibbs_sample(
        self,
        initial_state: np.ndarray,
        n_steps: int = 5,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Run Gibbs sampling to get model equilibrium state.

        Each step updates one NT dimension based on conditional distribution.

        Args:
            initial_state: Starting NT state
            n_steps: Number of Gibbs steps
            temperature: Sampling temperature (higher = more stochastic)

        Returns:
            Sampled model state
        """
        state = initial_state.copy()

        for _ in range(n_steps):
            for i in range(6):
                # Compute energy contribution from other NTs
                other_contribution = 0.0
                for j in range(6):
                    if i != j:
                        activated_j = 1 / (1 + np.exp(-5 * (state[j] - 0.5)))
                        other_contribution += self.K[j, i] * activated_j

                # Compute mean field update
                mean = 0.5 + other_contribution / 5.0  # Scale back from sigmoid

                # Sample with noise
                noise = np.random.randn() * temperature * 0.1
                state[i] = np.clip(mean + noise, 0.0, 1.0)

        return state

    def compute_energy(self, state: np.ndarray) -> float:
        """
        Compute energy of NT state under current coupling.

        Energy function (Hopfield-like):
            E = -0.5 * Σ_ij K_ij * s_i * s_j

        Lower energy = more stable/probable state.

        Args:
            state: NT concentration state [6,]

        Returns:
            Energy value (lower is more stable)
        """
        state = np.asarray(state).flatten()
        activated = 1 / (1 + np.exp(-5 * (state - 0.5)))

        # Quadratic energy
        energy = -0.5 * activated @ self.K @ activated

        return float(energy)

    def _enforce_ei_balance(self) -> None:
        """
        Enforce excitation/inhibition balance.

        GABA (inhibitory) and Glutamate (excitatory) should
        maintain homeostatic balance.
        """
        # GABA-Glu coupling should be negative (mutual inhibition)
        # Indices: GABA=4, Glu=5
        gaba_glu = self.K[4, 5]  # GABA's effect on Glu
        glu_gaba = self.K[5, 4]  # Glu's effect on GABA

        # Ensure these are inhibitory
        self.K[4, 5] = min(gaba_glu, -0.1)
        self.K[5, 4] = min(glu_gaba, -0.1)

    def get_coupling_strength(self) -> float:
        """Get overall coupling strength (Frobenius norm)."""
        return float(np.linalg.norm(self.K))

    # -------------------------------------------------------------------------
    # P2.3: Synaptic Scaling (Homeostatic Plasticity)
    # -------------------------------------------------------------------------

    def apply_synaptic_scaling(
        self,
        current_activity: np.ndarray,
        target_activity: float = 0.5,
        scaling_rate: float = 0.01,
    ) -> float:
        """
        Apply homeostatic synaptic scaling to maintain stable activity.

        Biological basis:
        - Turrigiano (2008): Synaptic scaling maintains network stability
        - All synapses scale multiplicatively to match target firing rate
        - Works over hours/days in biology, faster in simulation

        If activity is too high, scale down all coupling strengths.
        If activity is too low, scale up.

        Scaling factor: s = (target / current)^scaling_rate

        Args:
            current_activity: Current mean NT activity levels [6,]
            target_activity: Target mean activity level
            scaling_rate: How quickly to scale (0.01 = slow, 0.1 = fast)

        Returns:
            Scaling factor applied
        """
        current_activity = np.asarray(current_activity).flatten()
        mean_activity = float(np.mean(current_activity))

        if mean_activity < 0.01:
            mean_activity = 0.01  # Prevent division by zero

        # Compute scaling factor
        if mean_activity > target_activity:
            # Activity too high -> scale down coupling
            ratio = target_activity / mean_activity
            scale = 1.0 - scaling_rate * (1.0 - ratio)
        else:
            # Activity too low -> scale up coupling
            ratio = target_activity / mean_activity
            scale = 1.0 + scaling_rate * (ratio - 1.0)

        # Clamp scaling to reasonable range
        scale = np.clip(scale, 0.9, 1.1)

        # Apply multiplicative scaling
        self.K *= scale

        # Re-enforce bounds
        if self.config.enforce_bounds:
            self.K = self.bounds.clamp(self.K)

        return float(scale)

    def normalize_coupling(
        self,
        target_norm: float | None = None,
        method: str = "frobenius",
    ) -> float:
        """
        Normalize coupling matrix to target norm.

        This is a form of synaptic scaling that maintains total synaptic
        strength while allowing individual weights to change.

        Args:
            target_norm: Target Frobenius norm (None = use current as reference)
            method: "frobenius" or "spectral" (largest eigenvalue)

        Returns:
            Scaling factor applied
        """
        if method == "frobenius":
            current_norm = np.linalg.norm(self.K)
        elif method == "spectral":
            current_norm = np.abs(np.linalg.eigvals(self.K)).max()
        else:
            current_norm = np.linalg.norm(self.K)

        if current_norm < 1e-8:
            return 1.0

        if target_norm is None:
            # Use initial norm as target
            if not hasattr(self, '_initial_norm'):
                self._initial_norm = current_norm
            target_norm = self._initial_norm

        scale = target_norm / current_norm
        self.K *= scale

        # Re-enforce bounds
        if self.config.enforce_bounds:
            self.K = self.bounds.clamp(self.K)

        return float(scale)

    def get_stability_metrics(self) -> dict:
        """
        Get metrics related to coupling stability.

        Includes spectral radius, condition number, etc.
        """
        eigenvalues = np.linalg.eigvals(self.K)

        return {
            "spectral_radius": float(np.abs(eigenvalues).max()),
            "min_eigenvalue": float(np.min(np.real(eigenvalues))),
            "max_eigenvalue": float(np.max(np.real(eigenvalues))),
            "condition_number": float(np.linalg.cond(self.K)),
            "frobenius_norm": float(np.linalg.norm(self.K)),
            "is_stable": float(np.abs(eigenvalues).max()) < 1.0,
        }

    # -------------------------------------------------------------------------
    # P2.4: Attractor Stability Constraints (Jacobian Analysis)
    # -------------------------------------------------------------------------

    def compute_jacobian(
        self,
        state: np.ndarray,
        tau: float = 1.0,
    ) -> np.ndarray:
        """
        Compute Jacobian of NT dynamics at a given state.

        Biological basis:
        - Attractor networks must have stable fixed points
        - Jacobian eigenvalues determine local stability
        - All eigenvalues with negative real parts = stable attractor

        For dynamics: du_i/dt = -u_i/tau + sum_j K_ji * sigma(u_j)

        Jacobian: J_ij = d(du_i/dt)/du_j = -delta_ij/tau + K_ji * sigma'(u_j)

        where sigma'(u) = 5 * sigma(u) * (1 - sigma(u)) for our sigmoid.

        Args:
            state: NT concentration state [6,]
            tau: Time constant for decay

        Returns:
            Jacobian matrix [6, 6]
        """
        state = np.asarray(state).flatten()
        n = len(state)

        # Compute sigmoid and its derivative
        sigmoid = 1 / (1 + np.exp(-5 * (state - 0.5)))
        sigmoid_deriv = 5 * sigmoid * (1 - sigmoid)  # d sigma / du

        # Build Jacobian
        J = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: decay term + self-coupling
                    J[i, j] = -1.0 / tau + self.K[j, i] * sigmoid_deriv[j]
                else:
                    # Off-diagonal: coupling contribution
                    J[i, j] = self.K[j, i] * sigmoid_deriv[j]

        return J

    def analyze_attractor_stability(
        self,
        state: np.ndarray,
        tau: float = 1.0,
    ) -> dict:
        """
        Analyze stability of an attractor at the given state.

        Uses Jacobian eigenvalue analysis:
        - Stable if all eigenvalues have negative real parts
        - Margin of stability = distance from zero
        - Lyapunov characteristic indicates convergence rate

        Args:
            state: Fixed point or attractor state [6,]
            tau: Time constant

        Returns:
            Dictionary with stability analysis
        """
        J = self.compute_jacobian(state, tau)
        eigenvalues = np.linalg.eigvals(J)

        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        max_real = float(np.max(real_parts))
        min_real = float(np.min(real_parts))

        # Stable if all eigenvalues have negative real parts
        is_stable = max_real < 0

        # Stability margin: how far from instability
        stability_margin = -max_real if is_stable else max_real

        # Dominant eigenvalue determines convergence/divergence rate
        dominant_idx = np.argmax(real_parts)
        dominant_eigenvalue = eigenvalues[dominant_idx]

        # Check for oscillatory behavior (complex eigenvalues)
        has_oscillations = np.any(np.abs(imag_parts) > 1e-6)
        oscillation_freq = float(np.max(np.abs(imag_parts))) if has_oscillations else 0.0

        return {
            "is_stable": is_stable,
            "stability_margin": stability_margin,
            "max_real_eigenvalue": max_real,
            "min_real_eigenvalue": min_real,
            "dominant_eigenvalue": complex(dominant_eigenvalue),
            "has_oscillations": has_oscillations,
            "oscillation_frequency": oscillation_freq,
            "eigenvalues": eigenvalues.tolist(),
            "jacobian": J.tolist(),
            "convergence_rate": -max_real if is_stable else None,
        }

    def find_fixed_points(
        self,
        n_samples: int = 10,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        tau: float = 1.0,
    ) -> list[dict]:
        """
        Find fixed points of the NT dynamics through numerical search.

        Uses gradient descent to find states where du/dt ≈ 0.

        Args:
            n_samples: Number of random starting points
            max_iter: Maximum iterations per starting point
            tolerance: Convergence tolerance
            tau: Time constant

        Returns:
            List of found fixed points with stability info
        """
        fixed_points = []
        seen_states = []

        for _ in range(n_samples):
            # Random initial state
            state = np.random.uniform(0.1, 0.9, 6).astype(np.float32)

            for _ in range(max_iter):
                # Compute dynamics: du/dt = -u/tau + K^T @ sigma(u)
                sigmoid = 1 / (1 + np.exp(-5 * (state - 0.5)))
                coupling = self.K.T @ sigmoid

                du_dt = -state / tau + coupling

                # Check convergence
                if np.linalg.norm(du_dt) < tolerance:
                    break

                # Euler step toward fixed point
                state = np.clip(state + 0.1 * du_dt, 0.0, 1.0)

            # Check if this is a new fixed point
            is_new = True
            for seen in seen_states:
                if np.linalg.norm(state - seen) < 0.1:
                    is_new = False
                    break

            if is_new and np.linalg.norm(du_dt) < tolerance * 10:
                seen_states.append(state.copy())

                # Analyze stability
                stability = self.analyze_attractor_stability(state, tau)

                fixed_points.append({
                    "state": state.tolist(),
                    "residual": float(np.linalg.norm(du_dt)),
                    **stability,
                })

        # Sort by stability (stable ones first)
        fixed_points.sort(key=lambda x: (not x["is_stable"], x["max_real_eigenvalue"]))

        return fixed_points

    def enforce_stability_constraints(
        self,
        state: np.ndarray,
        tau: float = 1.0,
        target_margin: float = 0.1,
        max_adjustment: float = 0.1,
    ) -> bool:
        """
        Modify coupling to ensure attractor stability at given state.

        Uses Jacobian analysis to determine if adjustment is needed,
        then scales coupling to push eigenvalues toward stability.

        Biological basis:
        - Homeostatic mechanisms ensure network stability
        - If attractor becomes unstable, reduce coupling strength
        - This is a form of metaplasticity

        Args:
            state: State to stabilize as attractor
            tau: Time constant
            target_margin: Desired stability margin (negative eigenvalue threshold)
            max_adjustment: Maximum coupling adjustment per call

        Returns:
            True if adjustment was made, False if already stable
        """
        analysis = self.analyze_attractor_stability(state, tau)

        if analysis["is_stable"] and analysis["stability_margin"] >= target_margin:
            return False  # Already stable with sufficient margin

        # Need to reduce coupling to increase stability
        # Scaling down K reduces the positive eigenvalue contributions
        max_real = analysis["max_real_eigenvalue"]

        if max_real >= 0:
            # Unstable - need stronger adjustment
            scale = 1.0 - max_adjustment
        else:
            # Stable but insufficient margin
            needed_reduction = target_margin - analysis["stability_margin"]
            scale = 1.0 - min(max_adjustment, needed_reduction * 0.5)

        scale = np.clip(scale, 1.0 - max_adjustment, 1.0)

        # Apply scaling
        self.K *= scale

        # Re-enforce bounds
        if self.config.enforce_bounds:
            self.K = self.bounds.clamp(self.K)

        logger.debug(
            f"Stability constraint: scaled K by {scale:.4f}, "
            f"max_real: {max_real:.4f} -> target_margin: {target_margin:.4f}"
        )

        return True

    def compute_lyapunov_exponent(
        self,
        trajectory: np.ndarray,
        dt: float = 0.1,
        tau: float = 1.0,
    ) -> float:
        """
        Estimate largest Lyapunov exponent from a trajectory.

        The Lyapunov exponent measures the rate of separation of
        infinitesimally close trajectories:
        - Positive = chaotic (trajectories diverge)
        - Negative = stable (trajectories converge)
        - Zero = marginally stable

        Uses the Jacobian method: λ ≈ (1/T) * sum(log|J eigenvalue|)

        Args:
            trajectory: Sequence of states [T, 6]
            dt: Time step between observations
            tau: Time constant

        Returns:
            Estimated largest Lyapunov exponent
        """
        trajectory = np.asarray(trajectory)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(1, -1)

        T = len(trajectory)
        if T < 2:
            return 0.0

        sum_log_eigenvalues = 0.0

        for t in range(T):
            J = self.compute_jacobian(trajectory[t], tau)
            eigenvalues = np.linalg.eigvals(J)
            max_eigenvalue = np.max(np.abs(eigenvalues))

            if max_eigenvalue > 0:
                sum_log_eigenvalues += np.log(max_eigenvalue)

        # Average over trajectory
        lyapunov = sum_log_eigenvalues / (T * dt)

        return float(lyapunov)

    def get_attractor_basin_size(
        self,
        attractor_state: np.ndarray,
        n_samples: int = 100,
        perturbation_scale: float = 0.3,
        convergence_steps: int = 50,
        tau: float = 1.0,
    ) -> float:
        """
        Estimate the basin of attraction size for a given attractor.

        Samples random perturbations and checks if they converge
        back to the attractor.

        Args:
            attractor_state: The attractor state
            n_samples: Number of perturbation samples
            perturbation_scale: Standard deviation of perturbations
            convergence_steps: Steps to run dynamics
            tau: Time constant

        Returns:
            Fraction of samples that converged back (0-1)
        """
        attractor_state = np.asarray(attractor_state).flatten()
        converged_count = 0

        for _ in range(n_samples):
            # Random perturbation
            perturbation = np.random.randn(6) * perturbation_scale
            state = np.clip(attractor_state + perturbation, 0.0, 1.0)

            # Run dynamics
            for _ in range(convergence_steps):
                sigmoid = 1 / (1 + np.exp(-5 * (state - 0.5)))
                coupling = self.K.T @ sigmoid
                du_dt = -state / tau + coupling
                state = np.clip(state + 0.1 * du_dt, 0.0, 1.0)

            # Check if converged back
            if np.linalg.norm(state - attractor_state) < 0.15:
                converged_count += 1

        return converged_count / n_samples

    def get_stats(self) -> dict:
        """Get coupling statistics."""
        return {
            "update_count": self._update_count,
            "coupling_norm": self.get_coupling_strength(),
            "bounds_valid": self.bounds.is_valid(self.K),
            "mean_coupling": float(np.mean(self.K)),
            "max_coupling": float(np.max(self.K)),
            "min_coupling": float(np.min(self.K)),
            "ei_balance": float(self.K[4, 5] + self.K[5, 4]),  # Should be negative
        }

    def save_state(self) -> dict:
        """Save coupling state for persistence."""
        return {
            "K": self.K.tolist(),
            "update_count": self._update_count,
        }

    def load_state(self, state: dict) -> None:
        """Load coupling state from persistence."""
        self.K = np.array(state["K"], dtype=np.float32)
        self._update_count = state.get("update_count", 0)

        if self.config.enforce_bounds:
            self.K = self.bounds.clamp(self.K)


__all__ = [
    "LearnableCoupling",
    "BiologicalBounds",
    "CouplingConfig",
]
