"""
Active Inference and Predictive Coding Hierarchy.

Biological Basis:
- Predictive coding: Brain minimizes prediction error hierarchically
- Free energy principle: Action and perception minimize variational free energy
- Precision weighting: Attention as precision optimization

Mathematical Formulation (Friston 2005, Bogacz 2017):

Level Dynamics:
    dmu_l/dt = D_mu * [-mu_l + f(mu_{l+1}) + Sigma_l^-1 * epsilon_l]
    epsilon_l = x_{l-1} - g(mu_l)  # Prediction error at level l

Precision Dynamics (adaptive gain):
    dPi_l/dt = D_pi * [-Pi_l + Pi_prior + expected_precision(epsilon_l)]

Free Energy:
    F = sum_l [0.5 * epsilon_l^T * Pi_l * epsilon_l + 0.5 * log|Sigma_l|]

Action Selection:
    a* = argmin_a G(a) where G = expected free energy under action

References:
- Rao & Ballard (1999): Predictive coding in visual cortex
- Friston (2005): Theory of cortical responses
- Bogacz (2017): Tutorial on free-energy framework
- Parr & Friston (2019): Active inference and epistemic value
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActiveInferenceConfig:
    """Configuration for active inference hierarchy.

    Attributes:
        n_levels: Number of hierarchical levels
        dims: Dimensions at each level [sensory, level1, ...]
        precision_lr: Learning rate for precision
        belief_lr: Learning rate for beliefs
        free_energy_threshold: Threshold for action trigger
        temporal_horizon: Planning horizon for expected free energy
        precision_prior: Prior precision value
        action_dim: Dimension of action space
    """
    n_levels: int = 4
    dims: list[int] = field(default_factory=lambda: [1024, 512, 256, 128])
    precision_lr: float = 0.01
    belief_lr: float = 0.005
    free_energy_threshold: float = 0.1
    temporal_horizon: int = 5
    precision_prior: float = 1.0
    action_dim: int = 32


@dataclass
class BeliefState:
    """State of beliefs at one hierarchical level.

    Attributes:
        mean: mu - current belief (expected state)
        precision: Pi - confidence in belief
        prediction: g(mu) - prediction for level below
        error: epsilon - prediction error from level below
        free_energy: Local free energy contribution
    """
    mean: np.ndarray           # [dim]
    precision: np.ndarray      # [dim] or [dim, dim]
    prediction: np.ndarray     # [dim_below]
    error: np.ndarray          # [dim_below]
    free_energy: float = 0.0


class PrecisionWeightedLevel:
    """
    Single level in the predictive coding hierarchy.

    Implements precision-weighted prediction error minimization.

    Dynamics:
    - Beliefs (mu) update based on prediction errors from below
    - Predictions (g(mu)) are sent downward
    - Precision (Pi) adapts based on prediction error magnitude
    """

    def __init__(
        self,
        dim: int,
        dim_below: int,
        precision_prior: float = 1.0,
        learning_rate: float = 0.01,
    ):
        """
        Initialize predictive coding level.

        Args:
            dim: Dimension of this level's beliefs
            dim_below: Dimension of level below (input/prediction target)
            precision_prior: Prior on precision
            learning_rate: Learning rate for updates
        """
        self.dim = dim
        self.dim_below = dim_below
        self.precision_prior = precision_prior
        self.learning_rate = learning_rate

        # Generative model: mu -> prediction
        self.W_gen = np.random.randn(dim_below, dim).astype(np.float32) * 0.01

        # Recognition model: error -> mu update
        self.W_rec = np.random.randn(dim, dim_below).astype(np.float32) * 0.01

        # State
        self.mu = np.zeros(dim, dtype=np.float32)
        self.precision = np.ones(dim_below, dtype=np.float32) * precision_prior

        # Running estimates
        self._error_variance = np.ones(dim_below, dtype=np.float32)

        logger.debug(f"PrecisionWeightedLevel: {dim} -> {dim_below}")

    def compute_prediction(self) -> np.ndarray:
        """Generate prediction for level below: g(mu)."""
        # Nonlinear generative model
        prediction = np.tanh(self.W_gen @ self.mu)
        return prediction

    def compute_prediction_error(self, target: np.ndarray) -> np.ndarray:
        """
        Compute prediction error.

        epsilon = target - g(mu)

        Args:
            target: Input from level below or sensory

        Returns:
            Prediction error
        """
        prediction = self.compute_prediction()
        error = target - prediction
        return error

    def update_precision(self, error: np.ndarray) -> float:
        """
        Update precision based on prediction error.

        Precision ∝ 1 / expected_error^2

        Args:
            error: Current prediction error

        Returns:
            Mean precision value
        """
        # Update error variance estimate
        self._error_variance = (
            0.99 * self._error_variance +
            0.01 * (error ** 2)
        )

        # Precision = inverse variance (with regularization)
        self.precision = 1.0 / (self._error_variance + 0.1)

        # Soft clamp
        self.precision = np.clip(self.precision, 0.1, 10.0)

        return float(np.mean(self.precision))

    def compute_free_energy(self, error: np.ndarray) -> float:
        """
        Compute free energy contribution.

        F = 0.5 * epsilon^T * Pi * epsilon - 0.5 * log|Pi|

        Args:
            error: Prediction error

        Returns:
            Free energy
        """
        # Precision-weighted squared error
        weighted_error = 0.5 * np.sum(self.precision * error ** 2)

        # Complexity (negative log precision)
        complexity = -0.5 * np.sum(np.log(self.precision + 1e-8))

        return float(weighted_error + complexity)

    def belief_update(
        self,
        error_below: np.ndarray,
        pred_above: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Update beliefs based on prediction errors.

        dmu/dt = W_rec @ (Pi * error_below) + (pred_above - mu)

        Args:
            error_below: Precision-weighted error from level below
            pred_above: Prediction from level above (if any)

        Returns:
            Updated beliefs
        """
        # Bottom-up: recognize patterns from error
        weighted_error = self.precision * error_below
        bottom_up = self.W_rec @ weighted_error

        # Top-down: constraint from level above
        if pred_above is not None:
            top_down = pred_above - self.mu
        else:
            top_down = -self.mu * 0.01  # Regularization toward zero

        # Belief update
        dmu = self.learning_rate * (bottom_up + 0.5 * top_down)
        self.mu = self.mu + dmu

        return self.mu

    def get_state(self) -> BeliefState:
        """Get current belief state."""
        prediction = self.compute_prediction()
        return BeliefState(
            mean=self.mu.copy(),
            precision=self.precision.copy(),
            prediction=prediction,
            error=np.zeros(self.dim_below),  # Set by hierarchy
            free_energy=0.0,
        )


class ActiveInferenceHierarchy:
    """
    Hierarchical predictive coding with active inference.

    Implements the full free energy minimization framework:
    1. Perceptual inference: Update beliefs to explain sensory input
    2. Precision optimization: Adjust attention (precision weights)
    3. Active inference: Select actions to minimize expected free energy

    Architecture:
        Sensory -> Level 1 -> Level 2 -> ... -> Top Level
        ↓ errors up, predictions down ↓
    """

    def __init__(self, config: ActiveInferenceConfig | None = None):
        """
        Initialize active inference hierarchy.

        Args:
            config: Hierarchy configuration
        """
        self.config = config or ActiveInferenceConfig()

        # Create levels
        self.levels: list[PrecisionWeightedLevel] = []
        dims = self.config.dims

        for i in range(1, len(dims)):
            level = PrecisionWeightedLevel(
                dim=dims[i],
                dim_below=dims[i - 1],
                precision_prior=self.config.precision_prior,
                learning_rate=self.config.belief_lr,
            )
            self.levels.append(level)

        # Action model (for active inference)
        self.action_model = ActionModel(
            state_dim=dims[-1],
            action_dim=self.config.action_dim,
            horizon=self.config.temporal_horizon,
        )

        # Tracking
        self._total_free_energy = 0.0
        self._inference_steps = 0

        logger.info(
            f"ActiveInferenceHierarchy initialized: "
            f"{self.config.n_levels} levels, dims={dims}"
        )

    def infer(
        self,
        sensory: np.ndarray,
        n_iterations: int = 10,
    ) -> list[BeliefState]:
        """
        Run perceptual inference to explain sensory input.

        Iteratively updates beliefs at all levels to minimize
        prediction errors (variational inference).

        Args:
            sensory: Sensory input [sensory_dim]
            n_iterations: Number of belief update iterations

        Returns:
            List of BeliefState for each level
        """
        for _ in range(n_iterations):
            # Bottom-up pass: compute errors
            current_input = sensory
            errors = []

            for level in self.levels:
                error = level.compute_prediction_error(current_input)
                level.update_precision(error)
                errors.append(error)
                current_input = level.mu  # Pass beliefs up

            # Top-down pass: update beliefs
            for i in range(len(self.levels) - 1, -1, -1):
                # Get prediction from level above (if any)
                if i < len(self.levels) - 1:
                    pred_above = self.levels[i + 1].compute_prediction()
                else:
                    pred_above = None

                # Update beliefs
                self.levels[i].belief_update(errors[i], pred_above)

        # Compute free energy and return states
        self._total_free_energy = 0.0
        states = []

        current_input = sensory
        for i, level in enumerate(self.levels):
            error = level.compute_prediction_error(current_input)
            fe = level.compute_free_energy(error)
            self._total_free_energy += fe

            state = level.get_state()
            state.error = error
            state.free_energy = fe
            states.append(state)

            current_input = level.mu

        self._inference_steps += 1

        return states

    def learn(self) -> dict[str, float]:
        """
        Update generative and recognition models.

        Returns:
            Dict with learning statistics
        """
        total_update = 0.0

        for i, level in enumerate(self.levels):
            # Update generative model (reduce prediction error)
            # dW_gen = lr * error @ mu^T
            error = level.compute_prediction_error(
                level.mu if i == 0 else self.levels[i-1].mu
            )
            dW_gen = np.outer(error, level.mu) * level.learning_rate
            level.W_gen += dW_gen

            # Update recognition model (improve inference)
            # dW_rec = lr * mu @ error^T
            dW_rec = np.outer(level.mu, error * level.precision) * level.learning_rate
            level.W_rec += dW_rec

            total_update += np.linalg.norm(dW_gen) + np.linalg.norm(dW_rec)

        return {
            "total_update": float(total_update),
            "free_energy": self._total_free_energy,
            "inference_steps": self._inference_steps,
        }

    def compute_expected_free_energy(
        self,
        action: np.ndarray,
        current_states: list[BeliefState],
    ) -> float:
        """
        Compute expected free energy for an action.

        G = E[F(s')] + H[s'] - ambiguity - novelty

        This guides action selection in active inference.

        Args:
            action: Proposed action
            current_states: Current belief states

        Returns:
            Expected free energy
        """
        return self.action_model.compute_expected_free_energy(
            action,
            current_states[-1].mean,  # Top-level beliefs
            current_states[-1].precision,
        )

    def generate_action(
        self,
        goal_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate action that minimizes expected free energy.

        Args:
            goal_state: Optional goal state to reach

        Returns:
            Optimal action
        """
        top_beliefs = self.levels[-1].mu
        top_precision = self.levels[-1].precision

        return self.action_model.select_action(
            top_beliefs,
            top_precision,
            goal_state,
        )

    def get_total_free_energy(self) -> float:
        """Get total free energy across all levels."""
        return self._total_free_energy

    def get_stats(self) -> dict:
        """Get hierarchy statistics."""
        level_stats = []
        for i, level in enumerate(self.levels):
            level_stats.append({
                "dim": level.dim,
                "mean_precision": float(np.mean(level.precision)),
                "mean_belief": float(np.mean(np.abs(level.mu))),
            })

        return {
            "n_levels": len(self.levels),
            "total_free_energy": self._total_free_energy,
            "inference_steps": self._inference_steps,
            "levels": level_stats,
        }


class ActionModel:
    """
    Action selection model for active inference.

    Selects actions that minimize expected free energy:
    G = E[F] + H[s'] - epistemic value - pragmatic value
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 5,
    ):
        """
        Initialize action model.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            horizon: Planning horizon
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        # Transition model: s' = f(s, a)
        self.W_transition = np.random.randn(
            state_dim, state_dim + action_dim
        ).astype(np.float32) * 0.01

        # Goal preferences (desired outcomes)
        self.preferred_state = np.zeros(state_dim, dtype=np.float32)

        logger.debug(f"ActionModel: state_dim={state_dim}, action_dim={action_dim}")

    def predict_state(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Predict next state given current state and action."""
        combined = np.concatenate([current_state, action])
        next_state = np.tanh(self.W_transition @ combined)
        return next_state

    def compute_expected_free_energy(
        self,
        action: np.ndarray,
        current_state: np.ndarray,
        current_precision: np.ndarray,
    ) -> float:
        """
        Compute expected free energy for an action.

        G = risk + ambiguity
        risk = KL[Q(s') || P(s')] (divergence from preferences)
        ambiguity = expected uncertainty

        Args:
            action: Proposed action
            current_state: Current beliefs
            current_precision: Current precision

        Returns:
            Expected free energy
        """
        # Predict future state
        predicted_state = self.predict_state(current_state, action)

        # Risk: divergence from preferred state
        preference_error = predicted_state - self.preferred_state
        risk = 0.5 * np.sum(preference_error ** 2)

        # Ambiguity: expected uncertainty (inversely related to precision)
        mean_precision = np.mean(current_precision)
        ambiguity = 1.0 / (mean_precision + 0.1)

        # Epistemic value: information gain (simplified)
        epistemic = -0.1 * np.var(action)  # Encourage exploration

        return float(risk + ambiguity - epistemic)

    def select_action(
        self,
        current_state: np.ndarray,
        current_precision: np.ndarray,
        goal_state: np.ndarray | None = None,
        n_samples: int = 10,
    ) -> np.ndarray:
        """
        Select action minimizing expected free energy.

        Uses simple sampling-based optimization.

        Args:
            current_state: Current beliefs
            current_precision: Current precision
            goal_state: Optional goal (overrides preferred_state)
            n_samples: Number of action samples

        Returns:
            Best action
        """
        if goal_state is not None:
            self.preferred_state = goal_state

        best_action = None
        best_G = float('inf')

        for _ in range(n_samples):
            # Sample action
            action = np.random.randn(self.action_dim).astype(np.float32) * 0.5
            action = np.clip(action, -1, 1)

            # Compute expected free energy
            G = self.compute_expected_free_energy(
                action,
                current_state,
                current_precision,
            )

            if G < best_G:
                best_G = G
                best_action = action

        return best_action if best_action is not None else np.zeros(self.action_dim)


__all__ = [
    "ActiveInferenceConfig",
    "BeliefState",
    "PrecisionWeightedLevel",
    "ActiveInferenceHierarchy",
    "ActionModel",
]
