"""
Neural ODE Wrapper for Capsule Network Dynamics.

Biological Basis:
- Neural dynamics evolve continuously in time
- Capsule routing is an iterative settling process
- ODE formulation captures temporal evolution naturally

Mathematical Formulation:
    dz/dt = f_theta(z, t)

Where z = [activations, poses, routing] is the capsule state.

Capsule-specific dynamics:
    da_i/dt = -lambda * a_i + sigma(W_a * x + sum_j c_ij * v_j)
    dP_i/dt = -gamma * (P_i - I) + W_p * x + sum_j c_ij * T_ij * P_j
    dc_ij/dt = eta * (agreement(P_i, T_ij * P_j) - c_ij)

References:
- Chen et al. (2018): Neural ordinary differential equations
- Massaroli et al. (2020): Dissecting neural ODEs
- Sabour et al. (2017): Dynamic routing between capsules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


@dataclass
class NeuralODECapsuleConfig:
    """Configuration for Neural ODE capsule dynamics.

    Attributes:
        input_dim: Input embedding dimension
        num_capsules: Number of capsules
        capsule_dim: Dimension per capsule
        pose_dim: Pose matrix dimension (pose_dim x pose_dim)
        time_span: Integration time span (t0, t1)
        solver: ODE solver method
        rtol: Relative tolerance
        atol: Absolute tolerance
        activation_decay: Activation decay rate (lambda)
        pose_regularization: Pose regularization toward identity (gamma)
        routing_rate: Routing coefficient learning rate (eta)
        max_routing_iterations: Maximum routing iterations
    """
    input_dim: int = 1024
    num_capsules: int = 32
    capsule_dim: int = 16
    pose_dim: int = 4
    time_span: tuple[float, float] = (0.0, 1.0)
    solver: str = "RK45"  # RK45, RK23, DOP853, LSODA
    rtol: float = 1e-3
    atol: float = 1e-4
    activation_decay: float = 0.1
    pose_regularization: float = 0.01
    routing_rate: float = 0.5
    max_routing_iterations: int = 3


class CapsuleState:
    """State of the capsule network at a point in time.

    Contains:
    - activations: Capsule activation probabilities [num_capsules]
    - poses: Capsule pose matrices [num_capsules, pose_dim, pose_dim]
    - routing: Routing coefficients [num_capsules, num_capsules]
    """

    def __init__(
        self,
        activations: np.ndarray,
        poses: np.ndarray,
        routing: np.ndarray,
    ):
        """
        Initialize capsule state.

        Args:
            activations: Capsule activations [num_capsules]
            poses: Pose matrices [num_capsules, pose_dim, pose_dim]
            routing: Routing coefficients [num_capsules, num_capsules]
        """
        self.activations = activations
        self.poses = poses
        self.routing = routing

    @property
    def num_capsules(self) -> int:
        """Number of capsules."""
        return len(self.activations)

    @property
    def pose_dim(self) -> int:
        """Pose matrix dimension."""
        return self.poses.shape[1]

    def to_flat(self) -> np.ndarray:
        """Flatten state to 1D array for ODE solver."""
        return np.concatenate([
            self.activations.flatten(),
            self.poses.flatten(),
            self.routing.flatten(),
        ])

    @classmethod
    def from_flat(
        cls,
        flat: np.ndarray,
        num_capsules: int,
        pose_dim: int,
    ) -> CapsuleState:
        """Reconstruct state from flat array."""
        idx = 0

        # Activations
        activations = flat[idx:idx + num_capsules]
        idx += num_capsules

        # Poses
        pose_size = num_capsules * pose_dim * pose_dim
        poses = flat[idx:idx + pose_size].reshape(num_capsules, pose_dim, pose_dim)
        idx += pose_size

        # Routing
        routing = flat[idx:].reshape(num_capsules, num_capsules)

        return cls(activations, poses, routing)


class CapsuleODEFunc:
    """
    ODE function for capsule network dynamics.

    Implements the right-hand side of:
        dz/dt = f(z, t)

    where z is the capsule state (activations, poses, routing).
    """

    def __init__(
        self,
        config: NeuralODECapsuleConfig,
        input_transform: np.ndarray,
        pose_transform: np.ndarray,
    ):
        """
        Initialize ODE function.

        Args:
            config: Capsule configuration
            input_transform: Input to activation transform [num_caps, input_dim]
            pose_transform: Pose transformation matrices [num_caps, num_caps, pose, pose]
        """
        self.config = config
        self.W_a = input_transform  # [num_capsules, input_dim]
        self.T = pose_transform     # [num_capsules, num_capsules, pose, pose]

        # Current input (set before integration)
        self.current_input: np.ndarray | None = None

    def set_input(self, x: np.ndarray) -> None:
        """Set input for current forward pass."""
        self.current_input = x

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute state derivative.

        Args:
            t: Current time
            state: Flattened state vector

        Returns:
            State derivative (same shape as state)
        """
        return self.forward(t, state)

    def forward(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute capsule dynamics.

        Dynamics:
        da/dt = -lambda * a + sigma(W_a * x + routing contribution)
        dP/dt = -gamma * (P - I) + transformed pose contribution
        dc/dt = eta * (agreement - c)

        Args:
            t: Current time
            state: Flattened state vector

        Returns:
            State derivative
        """
        # Unpack state
        num_caps = self.config.num_capsules
        pose_dim = self.config.pose_dim

        cap_state = CapsuleState.from_flat(state, num_caps, pose_dim)
        a = cap_state.activations
        P = cap_state.poses
        c = cap_state.routing

        if self.current_input is None:
            raise ValueError("Input not set. Call set_input() before integration.")

        x = self.current_input

        # =====================================================
        # Activation dynamics: da/dt = -lambda * a + sigma(...)
        # =====================================================
        # Input contribution
        input_contrib = self.W_a @ x  # [num_caps]

        # Routing contribution: sum of weighted capsule votes
        routing_contrib = np.zeros(num_caps)
        for i in range(num_caps):
            for j in range(num_caps):
                # Vote magnitude from capsule j to i
                vote = a[j] * c[j, i] * np.linalg.norm(P[j])
                routing_contrib[i] += vote

        # Activation derivative
        activation_input = input_contrib + routing_contrib
        da = -self.config.activation_decay * a + self._squash(activation_input)

        # =====================================================
        # Pose dynamics: dP/dt = -gamma * (P - I) + transformation
        # =====================================================
        identity = np.eye(pose_dim)
        dP = np.zeros_like(P)

        for i in range(num_caps):
            # Regularization toward identity
            dP[i] = -self.config.pose_regularization * (P[i] - identity)

            # Transformation from other capsules
            for j in range(num_caps):
                if j != i:
                    # Transformed pose from j
                    T_ij = self.T[i, j]  # [pose, pose]
                    transformed = T_ij @ P[j]
                    dP[i] += c[j, i] * a[j] * (transformed - P[i]) * 0.1

        # =====================================================
        # Routing dynamics: dc/dt = eta * (agreement - c)
        # =====================================================
        dc = np.zeros_like(c)

        for i in range(num_caps):
            for j in range(num_caps):
                # Compute agreement between pose i and transformed pose from j
                T_ij = self.T[i, j]
                transformed_j = T_ij @ P[j]

                # Agreement as cosine similarity of flattened poses
                agreement = self._pose_agreement(P[i], transformed_j)

                # Routing update
                dc[j, i] = self.config.routing_rate * (agreement - c[j, i])

        # Pack derivatives
        d_state = CapsuleState(da, dP, dc)
        return d_state.to_flat()

    def _squash(self, x: np.ndarray) -> np.ndarray:
        """Squashing nonlinearity for capsule activations."""
        norm = np.abs(x) + 1e-8
        return (norm ** 2 / (1 + norm ** 2)) * np.sign(x)

    def _pose_agreement(self, P1: np.ndarray, P2: np.ndarray) -> float:
        """Compute agreement between two poses."""
        # Frobenius inner product, normalized
        dot = np.sum(P1 * P2)
        norm1 = np.linalg.norm(P1) + 1e-8
        norm2 = np.linalg.norm(P2) + 1e-8
        return dot / (norm1 * norm2)


class NeuralODECapsuleLayer:
    """
    Capsule layer with Neural ODE dynamics.

    Instead of discrete routing iterations, uses continuous-time
    ODE integration for capsule state evolution.

    Benefits:
    - More stable dynamics
    - Adaptive computation (solver chooses step size)
    - Reversible computation (adjoint method for gradients)
    - Natural temporal modeling
    """

    def __init__(self, config: NeuralODECapsuleConfig | None = None):
        """
        Initialize Neural ODE capsule layer.

        Args:
            config: Layer configuration
        """
        self.config = config or NeuralODECapsuleConfig()

        # Initialize transformations
        self._init_transforms()

        # Create ODE function
        self.ode_func = CapsuleODEFunc(
            self.config,
            self.W_a,
            self.T,
        )

        # Integration history (for analysis)
        self._trajectory_history: list = []

        logger.info(
            f"NeuralODECapsuleLayer initialized: "
            f"{self.config.num_capsules} capsules, "
            f"pose_dim={self.config.pose_dim}"
        )

    def _init_transforms(self):
        """Initialize transformation matrices."""
        num_caps = self.config.num_capsules
        input_dim = self.config.input_dim
        pose_dim = self.config.pose_dim

        # Input to activation
        self.W_a = np.random.randn(num_caps, input_dim).astype(np.float32) * 0.01

        # Pose transformations: T[i, j] transforms capsule j's pose for capsule i
        self.T = np.zeros(
            (num_caps, num_caps, pose_dim, pose_dim),
            dtype=np.float32
        )

        # Initialize as small perturbations of identity
        for i in range(num_caps):
            for j in range(num_caps):
                if i != j:
                    self.T[i, j] = (
                        np.eye(pose_dim) +
                        np.random.randn(pose_dim, pose_dim) * 0.1
                    )

    def _initial_state(self) -> CapsuleState:
        """Create initial capsule state."""
        num_caps = self.config.num_capsules
        pose_dim = self.config.pose_dim

        # Small random activations
        activations = np.random.uniform(0.01, 0.1, num_caps).astype(np.float32)

        # Poses start near identity
        poses = np.stack([
            np.eye(pose_dim) + np.random.randn(pose_dim, pose_dim) * 0.01
            for _ in range(num_caps)
        ]).astype(np.float32)

        # Uniform routing
        routing = np.ones((num_caps, num_caps), dtype=np.float32) / num_caps

        return CapsuleState(activations, poses, routing)

    def forward(
        self,
        x: np.ndarray,
        t_span: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass using ODE integration.

        Args:
            x: Input [input_dim]
            t_span: Integration time span (overrides config)

        Returns:
            Tuple of (activations, poses) at final time
        """
        t_span = t_span or self.config.time_span

        # Set input
        self.ode_func.set_input(x)

        # Initial state
        initial = self._initial_state()
        y0 = initial.to_flat()

        # Integrate ODE
        solution = solve_ivp(
            self.ode_func,
            t_span,
            y0,
            method=self.config.solver,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )

        if not solution.success:
            logger.warning(f"ODE integration failed: {solution.message}")

        # Extract final state
        final_state = CapsuleState.from_flat(
            solution.y[:, -1],
            self.config.num_capsules,
            self.config.pose_dim,
        )

        return final_state.activations, final_state.poses

    def get_trajectory(
        self,
        x: np.ndarray,
        n_points: int = 10,
    ) -> list[CapsuleState]:
        """
        Get full trajectory of capsule states over time.

        Args:
            x: Input [input_dim]
            n_points: Number of trajectory points

        Returns:
            List of CapsuleState at each time point
        """
        t_span = self.config.time_span
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # Set input
        self.ode_func.set_input(x)

        # Initial state
        initial = self._initial_state()
        y0 = initial.to_flat()

        # Integrate with evaluation points
        solution = solve_ivp(
            self.ode_func,
            t_span,
            y0,
            method=self.config.solver,
            t_eval=t_eval,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )

        # Convert to list of states
        trajectory = []
        for i in range(solution.y.shape[1]):
            state = CapsuleState.from_flat(
                solution.y[:, i],
                self.config.num_capsules,
                self.config.pose_dim,
            )
            trajectory.append(state)

        return trajectory

    def compute_energy(self, state: CapsuleState) -> float:
        """
        Compute energy of capsule state.

        Lower energy = more coherent representation.

        E = -sum_ij c_ij * a_i * a_j * agreement(P_i, T_ij @ P_j)

        Args:
            state: Capsule state

        Returns:
            Energy value
        """
        energy = 0.0
        num_caps = state.num_capsules

        for i in range(num_caps):
            for j in range(num_caps):
                if i != j:
                    T_ij = self.T[i, j]
                    transformed = T_ij @ state.poses[j]
                    agreement = self.ode_func._pose_agreement(
                        state.poses[i],
                        transformed
                    )
                    energy -= (
                        state.routing[j, i] *
                        state.activations[i] *
                        state.activations[j] *
                        agreement
                    )

        return float(energy)

    def get_stats(self) -> dict:
        """Get layer statistics."""
        return {
            "num_capsules": self.config.num_capsules,
            "pose_dim": self.config.pose_dim,
            "solver": self.config.solver,
            "time_span": self.config.time_span,
            "W_a_norm": float(np.linalg.norm(self.W_a)),
            "T_norm": float(np.linalg.norm(self.T)),
        }


__all__ = [
    "NeuralODECapsuleConfig",
    "CapsuleState",
    "CapsuleODEFunc",
    "NeuralODECapsuleLayer",
]
