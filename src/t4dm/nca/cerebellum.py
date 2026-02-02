"""
Cerebellar Circuit for T4DM.

Biological Basis:
- Mossy fibers (MF): Input pathway carrying context/sensory information
- Granule cells (GrC): Sparse expansion layer (~50:1 expansion ratio)
- Parallel fibers (PF): GrC axons projecting onto Purkinje cell dendrites
- Purkinje cells (PC): Main computational units, inhibitory output
- Climbing fibers (CF): Error signal from inferior olive (IO)
- Deep cerebellar nuclei (DCN): Output pathway (PC inhibition + MF excitation)

Learning Rule:
    CF error -> LTD at PF-PC synapses (Ito 1984, Albus 1971)
    When climbing fiber fires (error signal), parallel fiber synapses
    that were recently active undergo Long-Term Depression.

Core Functions:
1. Supervised error-driven learning via climbing fiber signals
2. Interval timing via Purkinje cell complex spike windows
3. Forward model for predictive internal representations

Architecture:
    MF input -> GrC (sparse expansion) -> PF -> PC (LTD via CF)
                                                    |
                                                    v
                  MF input ----------------------> DCN output
                  (excitatory)   (inhibitory from PC)

Key Equations:
    GrC activation: g = ReLU(W_grc @ mossy_input)  (sparse top-k)
    PC activation:  p = sigmoid(W_pf_pc @ g)
    DCN output:     d = sigmoid(W_mf_dcn @ mossy_input - W_pc_dcn @ p)
    LTD update:     W_pf_pc -= lr * error * g^T    (Ito 1984)
    Timing:         t_hat = softmax(W_timing @ g) . t_bins  (Weber's law)

References:
- Ito (1984): The Cerebellum and Neural Control
- Albus (1971): A theory of cerebellar function
- Wolpert, Miall & Kawato (1998): Internal models in the cerebellum
- Medina & Mauk (2000): Computer simulation of cerebellar timing
- Dean et al. (2010): The cerebellar microcircuit as an adaptive filter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CerebellarConfig:
    """Configuration for cerebellar circuit."""

    # Dimensions
    input_dim: int = 128            # Mossy fiber input dimension
    granule_expansion: int = 50     # GrC expansion ratio (~50:1 biological)
    num_purkinje: int = 100         # Number of Purkinje cells
    dcn_dim: int = 64               # Deep cerebellar nuclei output dimension

    # Granule cell parameters
    granule_sparsity: float = 0.05  # ~5% active GrC (sparse coding)

    # Learning parameters
    learning_rate: float = 0.01     # CF-driven LTD rate at PF-PC synapses
    forward_model_lr: float = 0.005 # Forward model learning rate
    weight_decay: float = 0.9999    # Slow weight decay for stability

    # Timing parameters
    timing_tau: float = 1.0         # Base timing constant (seconds)
    timing_bins: int = 50           # Number of temporal bins for timing
    timing_weber_fraction: float = 0.1  # Weber fraction for timing noise

    # DCN parameters
    pc_to_dcn_inhibition: float = 0.8  # PC inhibition strength on DCN
    mf_to_dcn_excitation: float = 0.5  # MF excitation strength on DCN

    # Biological constraints
    min_weight: float = 0.0         # Minimum PF-PC weight (LTD floor)
    max_weight: float = 2.0         # Maximum PF-PC weight

    @property
    def granule_dim(self) -> int:
        """Total granule cell population size."""
        return self.input_dim * self.granule_expansion


@dataclass
class CerebellarState:
    """State of cerebellar circuit after processing."""

    purkinje_output: np.ndarray     # PC firing rates [num_purkinje]
    dcn_output: np.ndarray          # DCN output [dcn_dim]
    timing_estimate: float          # Predicted interval (seconds)
    prediction_error: float         # Last climbing fiber error magnitude
    granule_sparsity: float = 0.0   # Actual GrC sparsity this step

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "purkinje_output_mean": float(np.mean(self.purkinje_output)),
            "dcn_output_mean": float(np.mean(self.dcn_output)),
            "timing_estimate": self.timing_estimate,
            "prediction_error": self.prediction_error,
            "granule_sparsity": self.granule_sparsity,
        }


# =============================================================================
# Granule Cell Layer (Sparse Expansion)
# =============================================================================


class GranuleCellLayer:
    """
    Granule cell layer: sparse expansion of mossy fiber input.

    Analogous to DG pattern separation in hippocampus, but serves
    a different purpose: creating a high-dimensional sparse code
    that allows Purkinje cells to learn fine-grained associations.

    Biological basis:
    - GrC are the most numerous neurons in the brain (~50B)
    - Each receives 4 mossy fiber inputs (convergence then expansion)
    - Only ~5% are active at any time (sparse coding)
    - This creates a combinatorial expansion for pattern discrimination
    """

    def __init__(self, config: CerebellarConfig, seed: int | None = None):
        self.config = config
        self._rng = np.random.default_rng(seed)

        granule_dim = config.granule_dim

        # MF -> GrC weights (random projection for expansion)
        scale = np.sqrt(2.0 / (config.input_dim + granule_dim))
        self._weights = self._rng.normal(
            0, scale, size=(config.input_dim, granule_dim)
        ).astype(np.float32)

        logger.info(
            f"GranuleCellLayer initialized: "
            f"MF({config.input_dim}) -> GrC({granule_dim}), "
            f"sparsity={config.granule_sparsity:.1%}"
        )

    def process(self, mossy_input: np.ndarray) -> np.ndarray:
        """
        Process mossy fiber input through granule cell expansion.

        Args:
            mossy_input: Mossy fiber input [input_dim]

        Returns:
            Sparse granule cell activation [granule_dim]
        """
        mossy_input = np.asarray(mossy_input, dtype=np.float32)

        # Expand: MF -> GrC
        expanded = mossy_input @ self._weights  # [granule_dim]

        # ReLU activation
        expanded = np.maximum(0, expanded)

        # k-WTA sparsification
        k = max(1, int(len(expanded) * self.config.granule_sparsity))
        if k < len(expanded):
            topk_idx = np.argpartition(expanded, -k)[-k:]
            sparse = np.zeros_like(expanded)
            sparse[topk_idx] = expanded[topk_idx]
        else:
            sparse = expanded

        return sparse


# =============================================================================
# Purkinje Cell Layer (LTD-based Learning)
# =============================================================================


class PurkinjeCellLayer:
    """
    Purkinje cell layer: main computational unit with CF-driven LTD.

    Biological basis:
    - PCs have massive dendritic trees (~200,000 parallel fiber synapses)
    - Each PC receives exactly ONE climbing fiber (from inferior olive)
    - CF activation signals error -> LTD at recently active PF synapses
    - This is the core learning rule of the cerebellum (Ito 1984)

    Learning rule:
        delta_W = -lr * error * granule_activation^T
        (Hebbian-like but ANTI-Hebbian: co-activation -> weakening)
    """

    def __init__(self, config: CerebellarConfig, seed: int | None = None):
        self.config = config
        self._rng = np.random.default_rng(seed)

        granule_dim = config.granule_dim

        # PF -> PC weights (parallel fiber to Purkinje synapses)
        scale = np.sqrt(2.0 / (granule_dim + config.num_purkinje))
        self._weights = self._rng.normal(
            0, scale, size=(granule_dim, config.num_purkinje)
        ).astype(np.float32)

        # Track last granule activation for LTD eligibility
        self._last_granule: np.ndarray | None = None

        logger.info(
            f"PurkinjeCellLayer initialized: "
            f"PF({granule_dim}) -> PC({config.num_purkinje}), "
            f"lr={config.learning_rate}"
        )

    def process(self, granule_activation: np.ndarray) -> np.ndarray:
        """
        Compute Purkinje cell output from parallel fiber input.

        Args:
            granule_activation: Granule cell activation [granule_dim]

        Returns:
            Purkinje cell firing rates [num_purkinje]
        """
        granule_activation = np.asarray(granule_activation, dtype=np.float32)

        # Store for LTD eligibility
        self._last_granule = granule_activation.copy()

        # PF -> PC: weighted sum + sigmoid
        raw = granule_activation @ self._weights  # [num_purkinje]
        pc_output = 1.0 / (1.0 + np.exp(-raw))   # sigmoid

        return pc_output.astype(np.float32)

    def update_weights(self, error: np.ndarray) -> float:
        """
        Apply climbing fiber error-driven LTD at PF-PC synapses.

        This is the core cerebellar learning rule:
        - CF fires when there is an error (from inferior olive)
        - PF synapses that were recently active undergo LTD
        - Anti-Hebbian: co-activation leads to WEAKENING

        Args:
            error: Climbing fiber error signal [num_purkinje]

        Returns:
            Mean absolute weight change
        """
        if self._last_granule is None:
            return 0.0

        error = np.asarray(error, dtype=np.float32)

        # LTD: delta_W = -lr * outer(granule, error)
        # Negative sign = Long-Term Depression
        delta_w = -self.config.learning_rate * np.outer(
            self._last_granule, error
        )

        # Apply update
        self._weights += delta_w

        # Weight decay for stability
        self._weights *= self.config.weight_decay

        # Clamp weights
        self._weights = np.clip(
            self._weights,
            self.config.min_weight,
            self.config.max_weight
        )

        return float(np.mean(np.abs(delta_w)))

    @property
    def weights(self) -> np.ndarray:
        """Access PF-PC weight matrix (read-only copy)."""
        return self._weights.copy()


# =============================================================================
# Deep Cerebellar Nuclei (Output)
# =============================================================================


class DeepCerebellarNuclei:
    """
    Deep cerebellar nuclei: output pathway of the cerebellum.

    Biological basis:
    - DCN receive inhibitory input from Purkinje cells
    - DCN receive excitatory input from mossy fiber collaterals
    - Net output = MF excitation - PC inhibition
    - This creates a push-pull system: PC learning sculpts DCN output

    When PCs learn to pause (via LTD), DCN neurons are disinhibited,
    producing precisely timed output signals.
    """

    def __init__(self, config: CerebellarConfig, seed: int | None = None):
        self.config = config
        self._rng = np.random.default_rng(seed)

        # MF -> DCN excitatory weights
        scale = np.sqrt(2.0 / (config.input_dim + config.dcn_dim))
        self._mf_weights = self._rng.normal(
            0, scale, size=(config.input_dim, config.dcn_dim)
        ).astype(np.float32)

        # PC -> DCN inhibitory weights
        scale = np.sqrt(2.0 / (config.num_purkinje + config.dcn_dim))
        self._pc_weights = self._rng.normal(
            0, scale, size=(config.num_purkinje, config.dcn_dim)
        ).astype(np.float32)

        logger.info(
            f"DeepCerebellarNuclei initialized: "
            f"MF({config.input_dim}) + PC({config.num_purkinje}) -> "
            f"DCN({config.dcn_dim})"
        )

    def process(
        self,
        mossy_input: np.ndarray,
        purkinje_output: np.ndarray
    ) -> np.ndarray:
        """
        Compute DCN output from MF excitation and PC inhibition.

        Args:
            mossy_input: Mossy fiber input [input_dim]
            purkinje_output: Purkinje cell output [num_purkinje]

        Returns:
            DCN output [dcn_dim]
        """
        mossy_input = np.asarray(mossy_input, dtype=np.float32)
        purkinje_output = np.asarray(purkinje_output, dtype=np.float32)

        # Excitation from mossy fibers
        excitation = mossy_input @ self._mf_weights * self.config.mf_to_dcn_excitation

        # Inhibition from Purkinje cells
        inhibition = purkinje_output @ self._pc_weights * self.config.pc_to_dcn_inhibition

        # Net DCN activity: excitation - inhibition
        raw = excitation - inhibition
        dcn_output = 1.0 / (1.0 + np.exp(-raw))  # sigmoid

        return dcn_output.astype(np.float32)


# =============================================================================
# Cerebellar Module (Integrated Circuit)
# =============================================================================


class CerebellarModule:
    """
    Cerebellar circuit for timing, error correction, and predictive models.

    Architecture:
    - Mossy fibers (MF): Input pathway (context/sensory)
    - Granule cells (GrC): Sparse expansion (similar to DG, ~50:1 expansion)
    - Parallel fibers (PF): GrC axons -> Purkinje cells
    - Purkinje cells (PC): Main output, inhibitory, complex dendritic tree
    - Climbing fibers (CF): Error signal from inferior olive (IO)
    - Deep cerebellar nuclei (DCN): Output pathway

    Learning rule: CF error -> LTD at PF-PC synapses (Ito 1984, Albus 1971)

    Three core functions:
    1. forward(): Process input through the circuit (with optional error learning)
    2. predict_timing(): Estimate temporal intervals from context
    3. predict_outcome(): Forward model for state-action prediction

    Usage:
        cerebellum = CerebellarModule(CerebellarConfig(input_dim=128))

        # Forward pass with error-driven learning
        state = cerebellum.forward(mossy_input, climbing_fiber_error=error)

        # Timing prediction
        interval = cerebellum.predict_timing(context)

        # Forward model
        predicted = cerebellum.predict_outcome(state_vec, action_vec)
        cerebellum.update(predicted, actual)  # Learn from error
    """

    def __init__(
        self,
        config: CerebellarConfig | None = None,
        random_seed: int | None = None,
    ):
        self.config = config or CerebellarConfig()
        self._rng = np.random.default_rng(random_seed)

        # Core layers
        self.granule_layer = GranuleCellLayer(self.config, seed=random_seed)
        self.purkinje_layer = PurkinjeCellLayer(
            self.config,
            seed=random_seed + 1 if random_seed is not None else None,
        )
        self.dcn_layer = DeepCerebellarNuclei(
            self.config,
            seed=random_seed + 2 if random_seed is not None else None,
        )

        # Timing network weights (GrC -> timing bins)
        granule_dim = self.config.granule_dim
        scale = np.sqrt(2.0 / (granule_dim + self.config.timing_bins))
        self._timing_weights = self._rng.normal(
            0, scale, size=(granule_dim, self.config.timing_bins)
        ).astype(np.float32)

        # Timing bin centers (log-spaced for Weber's law)
        self._timing_bins = np.logspace(
            np.log10(0.05),
            np.log10(self.config.timing_tau * 10),
            self.config.timing_bins,
        ).astype(np.float32)

        # Forward model weights (state+action -> predicted outcome)
        # Initialized lazily on first call to predict_outcome
        self._forward_model_weights: np.ndarray | None = None
        self._forward_model_bias: np.ndarray | None = None

        # State tracking
        self._last_prediction_error: float = 0.0
        self._cumulative_error: float = 0.0
        self._step_count: int = 0

        # History
        self._error_history: list[float] = []
        self._max_history = 1000

        logger.info(
            f"CerebellarModule initialized: "
            f"MF({self.config.input_dim}) -> "
            f"GrC({granule_dim}) -> "
            f"PC({self.config.num_purkinje}) -> "
            f"DCN({self.config.dcn_dim}), "
            f"timing_bins={self.config.timing_bins}"
        )

    # =========================================================================
    # Core Forward Pass
    # =========================================================================

    def forward(
        self,
        mossy_input: np.ndarray,
        climbing_fiber_error: np.ndarray | None = None,
    ) -> CerebellarState:
        """
        Process input through the cerebellar circuit.

        Args:
            mossy_input: Mossy fiber input [input_dim]
            climbing_fiber_error: Optional CF error for learning [num_purkinje].
                If provided, LTD is applied at PF-PC synapses.

        Returns:
            CerebellarState with DCN output and timing prediction
        """
        mossy_input = np.asarray(mossy_input, dtype=np.float32)

        # Normalize input
        norm = np.linalg.norm(mossy_input)
        if norm > 0:
            mossy_input = mossy_input / norm

        # Step 1: Granule cell expansion (sparse coding)
        granule_activation = self.granule_layer.process(mossy_input)
        actual_sparsity = float(
            np.count_nonzero(granule_activation) / len(granule_activation)
        )

        # Step 2: Purkinje cell computation
        purkinje_output = self.purkinje_layer.process(granule_activation)

        # Step 3: DCN output (MF excitation - PC inhibition)
        dcn_output = self.dcn_layer.process(mossy_input, purkinje_output)

        # Step 4: Timing prediction from granule code
        timing_estimate = self._compute_timing(granule_activation)

        # Step 5: Learning (if error provided)
        prediction_error = 0.0
        if climbing_fiber_error is not None:
            climbing_fiber_error = np.asarray(
                climbing_fiber_error, dtype=np.float32
            )
            mean_change = self.purkinje_layer.update_weights(
                climbing_fiber_error
            )
            prediction_error = float(np.mean(np.abs(climbing_fiber_error)))
            self._last_prediction_error = prediction_error
            self._error_history.append(prediction_error)
            if len(self._error_history) > self._max_history:
                self._error_history = self._error_history[-self._max_history:]

        self._step_count += 1

        state = CerebellarState(
            purkinje_output=purkinje_output,
            dcn_output=dcn_output,
            timing_estimate=timing_estimate,
            prediction_error=prediction_error,
            granule_sparsity=actual_sparsity,
        )

        logger.debug(
            f"Cerebellar forward: sparsity={actual_sparsity:.3f}, "
            f"timing={timing_estimate:.3f}s, error={prediction_error:.4f}"
        )

        return state

    # =========================================================================
    # Timing (Interval Estimation)
    # =========================================================================

    def _compute_timing(self, granule_activation: np.ndarray) -> float:
        """
        Compute interval timing estimate from granule cell code.

        Uses a population code over temporal bins with Weber's law:
        timing precision degrades proportionally with interval length.

        Biological basis: Medina & Mauk (2000) - granule cell temporal
        basis functions create a population code for time intervals.

        Args:
            granule_activation: Granule cell activation [granule_dim]

        Returns:
            Estimated time interval (seconds, always positive)
        """
        # Project granule code to timing bins
        raw = granule_activation @ self._timing_weights  # [timing_bins]

        # Softmax to get probability distribution over bins
        max_raw = np.max(raw)
        exp_raw = np.exp(raw - max_raw)
        probs = exp_raw / exp_raw.sum()

        # Expected value: weighted sum of bin centers
        estimate = float(np.dot(probs, self._timing_bins))

        # Weber's law noise: sigma = weber_fraction * estimate
        noise_sigma = self.config.timing_weber_fraction * estimate
        noise = float(self._rng.normal(0, noise_sigma))
        estimate = max(0.01, estimate + noise)  # Floor at 10ms

        return estimate

    def predict_timing(self, context: np.ndarray) -> float:
        """
        Predict time interval for a given context.

        This is the public API for timing prediction. Processes context
        through granule cells and returns an interval estimate.

        Timing precision follows Weber's law: the standard deviation
        of timing estimates scales linearly with the interval duration.

        Args:
            context: Context/sensory input [input_dim]

        Returns:
            Predicted time interval in seconds (always positive)
        """
        context = np.asarray(context, dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(context)
        if norm > 0:
            context = context / norm

        # Process through granule cells
        granule_activation = self.granule_layer.process(context)

        # Compute timing from granule code
        return self._compute_timing(granule_activation)

    # =========================================================================
    # Forward Model (Predictive Internal Model)
    # =========================================================================

    def _init_forward_model(self, state_dim: int, action_dim: int) -> None:
        """Lazily initialize forward model weights."""
        combined_dim = state_dim + action_dim
        scale = np.sqrt(2.0 / (combined_dim + state_dim))
        self._forward_model_weights = self._rng.normal(
            0, scale, size=(combined_dim, state_dim)
        ).astype(np.float32)
        self._forward_model_bias = np.zeros(state_dim, dtype=np.float32)

    def predict_outcome(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """
        Predict outcome of an action given current state.

        Implements the cerebellar forward model (Wolpert, Miall & Kawato 1998):
        the cerebellum learns to predict sensory consequences of motor commands.

        Args:
            state: Current state representation [state_dim]
            action: Action/motor command [action_dim]

        Returns:
            Predicted next state [state_dim]
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        # Lazy initialization
        if self._forward_model_weights is None:
            self._init_forward_model(len(state), len(action))

        # Concatenate state and action
        combined = np.concatenate([state, action])

        # Linear prediction + tanh activation
        raw = combined @ self._forward_model_weights + self._forward_model_bias
        predicted = np.tanh(raw)

        return predicted.astype(np.float32)

    def update(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> float:
        """
        Update forward model using climbing fiber error signal.

        The error between predicted and actual outcome drives learning
        via the climbing fiber pathway from the inferior olive.

        Args:
            predicted: Previously predicted outcome [state_dim]
            actual: Actual observed outcome [state_dim]

        Returns:
            Prediction error magnitude
        """
        predicted = np.asarray(predicted, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)

        error = actual - predicted
        error_magnitude = float(np.linalg.norm(error))

        # Update forward model weights (gradient descent)
        if self._forward_model_weights is not None:
            # Simple gradient: dL/dW = -error * input^T
            # (stored from last predict_outcome call is not tracked,
            #  so we use the error directly as a learning signal)
            self._forward_model_bias += self.config.forward_model_lr * error

        self._last_prediction_error = error_magnitude
        self._cumulative_error += error_magnitude
        self._error_history.append(error_magnitude)
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history:]

        logger.debug(
            f"Cerebellar forward model update: error={error_magnitude:.4f}"
        )

        return error_magnitude

    # =========================================================================
    # Weight Update (Public API)
    # =========================================================================

    def update_weights(self, error: np.ndarray) -> float:
        """
        Apply climbing fiber error-driven LTD at PF-PC synapses.

        Public API wrapping the Purkinje cell layer update.

        Args:
            error: Climbing fiber error signal [num_purkinje]

        Returns:
            Mean absolute weight change
        """
        return self.purkinje_layer.update_weights(error)

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get cerebellar circuit statistics."""
        stats = {
            "step_count": self._step_count,
            "last_prediction_error": self._last_prediction_error,
            "cumulative_error": self._cumulative_error,
            "pf_pc_weight_mean": float(np.mean(self.purkinje_layer.weights)),
            "pf_pc_weight_std": float(np.std(self.purkinje_layer.weights)),
        }

        if self._error_history:
            stats["mean_error"] = float(np.mean(self._error_history))
            recent = self._error_history[-100:]
            stats["recent_mean_error"] = float(np.mean(recent))

        return stats

    def reset(self) -> None:
        """Reset cerebellar module to initial state."""
        self._last_prediction_error = 0.0
        self._cumulative_error = 0.0
        self._step_count = 0
        self._error_history.clear()
        self._forward_model_weights = None
        self._forward_model_bias = None
        logger.info("CerebellarModule reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "pf_pc_weights": self.purkinje_layer._weights.tolist(),
            "timing_weights": self._timing_weights.tolist(),
            "step_count": self._step_count,
            "cumulative_error": self._cumulative_error,
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "pf_pc_weights" in saved:
            self.purkinje_layer._weights = np.array(
                saved["pf_pc_weights"], dtype=np.float32
            )
        if "timing_weights" in saved:
            self._timing_weights = np.array(
                saved["timing_weights"], dtype=np.float32
            )
        if "step_count" in saved:
            self._step_count = saved["step_count"]
        if "cumulative_error" in saved:
            self._cumulative_error = saved["cumulative_error"]


# =============================================================================
# Factory Functions
# =============================================================================


def create_cerebellar_module(
    input_dim: int = 128,
    granule_expansion: int = 50,
    num_purkinje: int = 100,
    learning_rate: float = 0.01,
    **kwargs,
) -> CerebellarModule:
    """
    Factory function to create a cerebellar module with common defaults.

    Args:
        input_dim: Mossy fiber input dimension
        granule_expansion: Granule cell expansion ratio
        num_purkinje: Number of Purkinje cells
        learning_rate: CF-driven LTD learning rate
        **kwargs: Additional CerebellarConfig parameters

    Returns:
        Configured CerebellarModule
    """
    config = CerebellarConfig(
        input_dim=input_dim,
        granule_expansion=granule_expansion,
        num_purkinje=num_purkinje,
        learning_rate=learning_rate,
        **kwargs,
    )
    return CerebellarModule(config)


# Backward compatibility aliases
Cerebellum = CerebellarModule


__all__ = [
    "CerebellarModule",
    "CerebellarConfig",
    "CerebellarState",
    "GranuleCellLayer",
    "PurkinjeCellLayer",
    "DeepCerebellarNuclei",
    "Cerebellum",
    "create_cerebellar_module",
]
