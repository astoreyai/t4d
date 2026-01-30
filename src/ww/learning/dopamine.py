"""
Dopamine-like Reward Prediction Error for World Weaver.

Biological Basis:
- Dopamine neurons encode surprise: firing increases when reward > expected
- Learning is proportional to prediction error, not raw reward
- This prevents over-updating on expected outcomes
- Unexpected successes/failures drive adaptation

Implementation:
- Maintains per-memory expected value estimates
- Computes RPE (δ) = actual_outcome - expected_outcome
- Uses RPE to modulate learning rates across systems
- Updates expectations via exponential moving average

Integration Points:
1. Reconsolidation: learning_rate *= |RPE| (surprise-modulated updates)
2. Fusion training: target_rewards = RPE (learn from surprises, not raw outcomes)
3. Eligibility traces: TD error already uses this principle
4. Memory consolidation: prioritize surprising memories for replay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


class LearnedValueEstimator:
    """
    Neural network-based value function that generalizes across embeddings.

    Biological Basis:
    - The brain doesn't maintain explicit lookup tables of memory->value
    - Value is encoded in synaptic weights and reconstructed from representations
    - Similar memories should have similar expected values (generalization)

    Implementation:
    - Simple 2-layer MLP: embedding -> hidden -> value
    - Uses ReLU activation for biological plausibility (rectification)
    - Updates via gradient descent on TD error
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 256,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize learned value estimator.

        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for weight updates
            weight_decay: L2 regularization strength
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Xavier initialization for stable gradients
        self.W1 = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(2.0 / embedding_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

        # Cache for backprop
        self._last_embedding: np.ndarray | None = None
        self._last_hidden: np.ndarray | None = None
        self._last_output: float = 0.5

        # Statistics
        self._update_count = 0
        self._total_td_error = 0.0

    def estimate(self, embedding: np.ndarray) -> float:
        """
        Estimate value from embedding.

        Args:
            embedding: Memory embedding vector

        Returns:
            Estimated value [0, 1]
        """
        embedding = np.asarray(embedding).flatten()

        # Pad or truncate to expected dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        elif len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]

        # Forward pass
        hidden = np.maximum(0, embedding @ self.W1 + self.b1)  # ReLU
        output = float(hidden @ self.W2 + self.b2)

        # Sigmoid to bound [0, 1]
        value = 1.0 / (1.0 + np.exp(-np.clip(output, -10, 10)))

        # Cache for potential update
        self._last_embedding = embedding
        self._last_hidden = hidden
        self._last_output = value

        return value

    def update(self, embedding: np.ndarray, td_error: float) -> None:
        """
        Update value function weights using TD error.

        Uses semi-gradient TD learning:
        - Only backprop through value estimate, not target
        - Updates move prediction toward actual outcome

        Args:
            embedding: Memory embedding that was evaluated
            td_error: TD error (actual - predicted)
        """
        embedding = np.asarray(embedding).flatten()

        # Ensure correct dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        elif len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]

        # Forward pass (recompute to ensure fresh values)
        hidden = np.maximum(0, embedding @ self.W1 + self.b1)
        output = float(hidden @ self.W2 + self.b2)
        value = 1.0 / (1.0 + np.exp(-np.clip(output, -10, 10)))

        # Backward pass
        # d_loss/d_output = -td_error (we want to minimize (target - prediction)^2)
        # d_sigmoid/d_output = value * (1 - value)
        d_output = -td_error * value * (1 - value)

        # Gradient for W2, b2
        d_W2 = np.outer(hidden, d_output)
        d_b2 = d_output

        # Gradient for hidden layer
        d_hidden = (self.W2 * d_output).flatten()
        d_hidden = d_hidden * (hidden > 0)  # ReLU gradient

        # Gradient for W1, b1
        d_W1 = np.outer(embedding, d_hidden)
        d_b1 = d_hidden

        # Update with weight decay
        self.W2 -= self.learning_rate * (d_W2 + self.weight_decay * self.W2)
        self.b2 -= self.learning_rate * d_b2
        self.W1 -= self.learning_rate * (d_W1 + self.weight_decay * self.W1)
        self.b1 -= self.learning_rate * d_b1

        # Statistics
        self._update_count += 1
        self._total_td_error += abs(td_error)

    def get_stats(self) -> dict:
        """Get value estimator statistics."""
        return {
            "update_count": self._update_count,
            "avg_td_error": self._total_td_error / max(1, self._update_count),
            "weight_norm_W1": float(np.linalg.norm(self.W1)),
            "weight_norm_W2": float(np.linalg.norm(self.W2)),
        }

    def save_state(self) -> dict:
        """Save weights for persistence."""
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
        }

    def load_state(self, state: dict) -> None:
        """Load weights from saved state."""
        self.W1 = np.array(state["W1"])
        self.b1 = np.array(state["b1"])
        self.W2 = np.array(state["W2"])
        self.b2 = np.array(state["b2"])
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]


@dataclass
class RewardPredictionError:
    """Record of a dopamine-like prediction error."""

    memory_id: UUID
    expected: float
    actual: float
    rpe: float  # δ = actual - expected
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_positive_surprise(self) -> bool:
        """Better than expected."""
        return self.rpe > 0.05

    @property
    def is_negative_surprise(self) -> bool:
        """Worse than expected."""
        return self.rpe < -0.05

    @property
    def surprise_magnitude(self) -> float:
        """Absolute surprise for learning rate modulation."""
        return abs(self.rpe)


class DopamineSystem:
    """
    Reward prediction error system inspired by midbrain dopamine neurons.

    The brain's dopamine system doesn't signal raw reward - it signals
    unexpected reward. This is crucial for efficient learning:
    - Expected outcomes (δ≈0): minimal learning, don't waste updates
    - Positive surprise (δ>0): strengthen what led to this
    - Negative surprise (δ<0): weaken what led to this

    Formula: δ = r - V(m)
    - r = actual outcome
    - V(m) = expected value of memory m

    Two modes of operation:
    1. Lookup mode (legacy): Value estimates stored in dict per memory ID
    2. Learned mode (biological): Value estimated from embeddings via MLP

    The learned mode enables generalization - similar memories have similar
    expected values without explicit tracking.
    """

    def __init__(
        self,
        default_expected: float = 0.5,
        value_learning_rate: float = 0.1,
        surprise_threshold: float = 0.05,
        max_rpe_magnitude: float = 1.0,
        use_learned_values: bool = False,
        embedding_dim: int = 1024,
        td_lambda: float = 0.9,
        discount_gamma: float = 0.95,
    ):
        """
        Initialize dopamine system.

        Args:
            default_expected: Default expected value for new memories
            value_learning_rate: α for updating value estimates
            surprise_threshold: Minimum |δ| to count as surprising
            max_rpe_magnitude: Clip RPE magnitude for stability
            use_learned_values: Use MLP-based value estimation (generalizes across embeddings)
            embedding_dim: Dimension of embeddings (for learned mode)
            td_lambda: Eligibility trace decay rate for TD(λ)
            discount_gamma: Temporal discount factor for future rewards
        """
        self.default_expected = default_expected
        self.value_learning_rate = value_learning_rate
        self.surprise_threshold = surprise_threshold
        self.max_rpe_magnitude = max_rpe_magnitude
        self.use_learned_values = use_learned_values
        self.td_lambda = td_lambda
        self.discount_gamma = discount_gamma

        # Per-memory value estimates V(m) - lookup mode
        # MEM-004 FIX: Add size limit for memory protection
        self._value_estimates: dict[str, float] = {}
        self._max_tracked_memories = 100000

        # P2.2: TD(λ) eligibility traces per memory
        self._eligibility_traces: dict[str, float] = {}
        self._trace_decay = td_lambda * discount_gamma

        # Learned value estimator - biological mode
        self._value_network: LearnedValueEstimator | None = None
        if use_learned_values:
            self._value_network = LearnedValueEstimator(
                embedding_dim=embedding_dim,
                learning_rate=value_learning_rate,
            )
        # Embedding cache for learned mode
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._max_embedding_cache = 10000

        # Outcome counts for uncertainty estimation
        self._outcome_counts: dict[str, int] = {}

        # RPE history for analysis - bounded
        # MEM-004 FIX: Limit history size
        self._rpe_history: list[RewardPredictionError] = []
        self._max_history_size = 10000

    def get_expected_value(
        self,
        memory_id: UUID,
        embedding: np.ndarray | None = None,
    ) -> float:
        """
        Get expected outcome value for a memory.

        In learned mode, uses the embedding to estimate value (enables generalization).
        In lookup mode, uses stored per-memory estimates.

        Args:
            memory_id: Memory to get expectation for
            embedding: Memory embedding (required for learned mode, optional for lookup)

        Returns:
            Expected value [0, 1]
        """
        mem_id_str = str(memory_id)

        # Learned mode: use neural network
        if self.use_learned_values and self._value_network is not None:
            if embedding is not None:
                # Cache embedding for later updates
                self._embedding_cache[mem_id_str] = np.asarray(embedding)
                if len(self._embedding_cache) > self._max_embedding_cache:
                    # Remove oldest entries
                    keys_to_remove = list(self._embedding_cache.keys())[:-self._max_embedding_cache]
                    for key in keys_to_remove:
                        del self._embedding_cache[key]
                return self._value_network.estimate(embedding)
            elif mem_id_str in self._embedding_cache:
                return self._value_network.estimate(self._embedding_cache[mem_id_str])
            else:
                # No embedding available, fall back to default
                return self.default_expected

        # Lookup mode: use dict
        return self._value_estimates.get(mem_id_str, self.default_expected)

    def compute_rpe(
        self,
        memory_id: UUID,
        actual_outcome: float,
        embedding: np.ndarray | None = None,
    ) -> RewardPredictionError:
        """
        Compute reward prediction error for a memory.

        This is the dopamine signal: δ = actual - expected

        Args:
            memory_id: Memory that was retrieved/used
            actual_outcome: Observed outcome [0, 1]
            embedding: Memory embedding (for learned value mode)

        Returns:
            RewardPredictionError with δ and metadata
        """
        expected = self.get_expected_value(memory_id, embedding=embedding)
        rpe = actual_outcome - expected

        # Clip for stability
        rpe = np.clip(rpe, -self.max_rpe_magnitude, self.max_rpe_magnitude)

        result = RewardPredictionError(
            memory_id=memory_id,
            expected=expected,
            actual=actual_outcome,
            rpe=float(rpe)
        )

        self._rpe_history.append(result)

        # MEM-004 FIX: Trim history if over limit
        if len(self._rpe_history) > self._max_history_size:
            self._rpe_history = self._rpe_history[-self._max_history_size:]

        return result

    def batch_compute_rpe(
        self,
        memory_outcomes: dict[str, float]
    ) -> dict[str, RewardPredictionError]:
        """
        Compute RPE for multiple memories.

        Args:
            memory_outcomes: Memory ID -> actual outcome

        Returns:
            Memory ID -> RewardPredictionError
        """
        return {
            mem_id: self.compute_rpe(UUID(mem_id), outcome)
            for mem_id, outcome in memory_outcomes.items()
        }

    def update_expectations(
        self,
        memory_id: UUID,
        actual_outcome: float,
        embedding: np.ndarray | None = None,
    ) -> float:
        """
        Update value estimate based on observed outcome.

        In learned mode: Updates MLP weights via TD error backprop
        In lookup mode: Uses exponential moving average

        Args:
            memory_id: Memory to update
            actual_outcome: Observed outcome
            embedding: Memory embedding (for learned value mode)

        Returns:
            New expected value
        """
        mem_id_str = str(memory_id)

        # Learned mode: update neural network
        if self.use_learned_values and self._value_network is not None:
            # Get embedding from cache if not provided
            if embedding is None:
                embedding = self._embedding_cache.get(mem_id_str)

            if embedding is not None:
                current = self._value_network.estimate(embedding)
                td_error = actual_outcome - current
                self._value_network.update(embedding, td_error)
                new_value = self._value_network.estimate(embedding)
            else:
                # No embedding, can't learn
                new_value = self.default_expected

            # Track outcome count for uncertainty
            self._outcome_counts[mem_id_str] = self._outcome_counts.get(mem_id_str, 0) + 1
            return new_value

        # Lookup mode: EMA update
        current = self._value_estimates.get(mem_id_str, self.default_expected)
        new_value = current + self.value_learning_rate * (actual_outcome - current)
        self._value_estimates[mem_id_str] = new_value

        # Track outcome count for uncertainty
        self._outcome_counts[mem_id_str] = self._outcome_counts.get(mem_id_str, 0) + 1

        # MEM-004 FIX: Cleanup old entries if over limit
        if len(self._value_estimates) > self._max_tracked_memories:
            self._cleanup_old_memories()

        return new_value

    def _cleanup_old_memories(self) -> None:
        """Remove least-used memory entries to enforce size limit."""
        # Remove memories with fewest observations
        if len(self._value_estimates) <= self._max_tracked_memories:
            return

        # Sort by count and remove lowest
        sorted_memories = sorted(
            self._outcome_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        keep_ids = set(m[0] for m in sorted_memories[:self._max_tracked_memories])

        self._value_estimates = {k: v for k, v in self._value_estimates.items() if k in keep_ids}
        self._outcome_counts = {k: v for k, v in self._outcome_counts.items() if k in keep_ids}

    def batch_update_expectations(
        self,
        memory_outcomes: dict[str, float]
    ) -> dict[str, float]:
        """
        Batch update value estimates.

        Args:
            memory_outcomes: Memory ID -> actual outcome

        Returns:
            Memory ID -> new expected value
        """
        return {
            mem_id: self.update_expectations(UUID(mem_id), outcome)
            for mem_id, outcome in memory_outcomes.items()
        }

    # =========================================================================
    # P2.2: TD(λ) Multi-Step Credit Assignment
    # =========================================================================

    def mark_memory_active(self, memory_id: UUID) -> None:
        """
        Mark a memory as active (recently accessed/used).

        This sets its eligibility trace to 1.0, making it eligible
        for credit when a reward arrives.

        Args:
            memory_id: Memory to mark as active
        """
        mem_id_str = str(memory_id)
        self._eligibility_traces[mem_id_str] = 1.0

    def decay_eligibility_traces(self) -> None:
        """
        Decay all eligibility traces by λγ.

        Called each timestep to fade out old traces.
        Traces below threshold are removed.
        """
        threshold = 0.01
        to_remove = []

        for mem_id, trace in self._eligibility_traces.items():
            new_trace = trace * self._trace_decay
            if new_trace < threshold:
                to_remove.append(mem_id)
            else:
                self._eligibility_traces[mem_id] = new_trace

        for mem_id in to_remove:
            del self._eligibility_traces[mem_id]

    def update_with_td_lambda(
        self,
        td_error: float,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """
        Update all eligible memories using TD(λ).

        This is the key TD(λ) update: all memories with non-zero
        eligibility traces get updated proportional to their trace.

        Biological basis:
        - Eligibility traces represent "synaptic tags"
        - When dopamine arrives (TD error), tagged synapses are modified
        - This assigns credit to actions that led to reward

        Update rule: V(s) += α * δ * e(s)

        Args:
            td_error: Computed TD error (dopamine signal)
            embeddings: Optional dict of memory_id -> embedding

        Returns:
            Dict of memory_id -> updated value
        """
        updated_values = {}

        for mem_id_str, trace in self._eligibility_traces.items():
            if trace < 0.01:
                continue

            # Effective learning rate scaled by eligibility
            effective_lr = self.value_learning_rate * trace

            if self.use_learned_values and self._value_network is not None:
                # Get embedding
                embedding = None
                if embeddings and mem_id_str in embeddings:
                    embedding = embeddings[mem_id_str]
                elif mem_id_str in self._embedding_cache:
                    embedding = self._embedding_cache[mem_id_str]

                if embedding is not None:
                    # Update with trace-scaled TD error
                    self._value_network.update(embedding, td_error * trace)
                    new_value = self._value_network.estimate(embedding)
                    updated_values[mem_id_str] = new_value
            else:
                # Lookup mode update
                current = self._value_estimates.get(mem_id_str, self.default_expected)
                new_value = current + effective_lr * td_error
                new_value = float(np.clip(new_value, 0.0, 1.0))
                self._value_estimates[mem_id_str] = new_value
                updated_values[mem_id_str] = new_value

        return updated_values

    def process_reward_with_traces(
        self,
        reward: float,
        current_memory_id: UUID,
        next_memory_id: UUID | None = None,
        terminal: bool = False,
        current_embedding: np.ndarray | None = None,
        next_embedding: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Process a reward signal and update all eligible memories via TD(λ).

        This combines TD error computation with trace-based credit assignment.

        Args:
            reward: Immediate reward signal
            current_memory_id: Memory in current state
            next_memory_id: Memory in next state (if any)
            terminal: Whether this is a terminal state
            current_embedding: Embedding of current memory
            next_embedding: Embedding of next memory

        Returns:
            (td_error, updated_values_dict)
        """
        # Mark current memory as active
        self.mark_memory_active(current_memory_id)

        # Get value estimates
        v_current = self.get_expected_value(current_memory_id, current_embedding)
        if terminal:
            v_next = 0.0
        elif next_memory_id is not None:
            v_next = self.get_expected_value(next_memory_id, next_embedding)
        else:
            v_next = v_current  # Bootstrap

        # Compute TD error: δ = r + γV(s') - V(s)
        td_error = reward + self.discount_gamma * v_next - v_current
        td_error = float(np.clip(td_error, -self.max_rpe_magnitude, self.max_rpe_magnitude))

        # Build embeddings dict for update
        embeddings = {}
        if current_embedding is not None:
            embeddings[str(current_memory_id)] = current_embedding
        embeddings.update(self._embedding_cache)

        # Update all eligible memories
        updated = self.update_with_td_lambda(td_error, embeddings)

        # Decay traces for next timestep
        self.decay_eligibility_traces()

        return td_error, updated

    def get_eligibility_trace(self, memory_id: UUID) -> float:
        """
        Get current eligibility trace for a memory.

        Args:
            memory_id: Memory to check

        Returns:
            Eligibility value [0, 1]
        """
        return self._eligibility_traces.get(str(memory_id), 0.0)

    def clear_eligibility_traces(self) -> None:
        """Clear all eligibility traces (e.g., at episode end)."""
        self._eligibility_traces.clear()

    def get_uncertainty(self, memory_id: UUID) -> float:
        """
        Get uncertainty in value estimate (inverse of experience count).

        Fewer observations = higher uncertainty = larger learning rates.

        Args:
            memory_id: Memory to check

        Returns:
            Uncertainty factor [0, 1] where 1 = very uncertain
        """
        count = self._outcome_counts.get(str(memory_id), 0)
        # Uncertainty decreases with experience: 1/(1+count)
        return 1.0 / (1.0 + count)

    def modulate_learning_rate(
        self,
        base_lr: float,
        rpe: RewardPredictionError,
        use_uncertainty: bool = True
    ) -> float:
        """
        Compute surprise-modulated learning rate.

        Learning rate scales with:
        1. |RPE|: More surprise = more learning
        2. Uncertainty: More uncertain = more learning

        Args:
            base_lr: Base learning rate
            rpe: Computed reward prediction error
            use_uncertainty: Whether to boost LR for uncertain memories

        Returns:
            Modulated learning rate
        """
        # Surprise modulation: scale by |δ|, but ensure minimum learning
        surprise_factor = max(rpe.surprise_magnitude, 0.1)

        # Optional uncertainty boost for rarely-seen memories
        uncertainty_factor = 1.0
        if use_uncertainty:
            uncertainty = self.get_uncertainty(rpe.memory_id)
            uncertainty_factor = 1.0 + uncertainty  # Range [1, 2]

        return base_lr * surprise_factor * uncertainty_factor

    def get_rpe_for_fusion_training(
        self,
        memory_outcomes: dict[str, float]
    ) -> dict[str, float]:
        """
        Convert outcomes to RPE-based training targets.

        Instead of training fusion on raw outcomes, train on prediction errors.
        This teaches the system to identify surprising (informative) memories.

        Args:
            memory_outcomes: Memory ID -> actual outcome

        Returns:
            Memory ID -> RPE (shifted to [0, 1] for ranking)
        """
        rpes = self.batch_compute_rpe(memory_outcomes)

        # Shift RPE from [-1, 1] to [0, 1] for ranking loss
        # δ=+1 becomes 1.0 (best), δ=-1 becomes 0.0 (worst)
        return {
            mem_id: (rpe.rpe + 1.0) / 2.0
            for mem_id, rpe in rpes.items()
        }

    def get_stats(self) -> dict:
        """
        Get dopamine system statistics.

        Returns:
            Statistics dict for monitoring
        """
        base_stats = {
            "total_signals": 0,
            "positive_surprises": 0,
            "negative_surprises": 0,
            "avg_rpe": 0.0,
            "memories_tracked": len(self._value_estimates),
            "use_learned_values": self.use_learned_values,
            "config": self.get_config(),
        }

        if not self._rpe_history:
            return base_stats

        positive = [r for r in self._rpe_history if r.is_positive_surprise]
        negative = [r for r in self._rpe_history if r.is_negative_surprise]

        stats = {
            "total_signals": len(self._rpe_history),
            "positive_surprises": len(positive),
            "negative_surprises": len(negative),
            "avg_rpe": float(np.mean([r.rpe for r in self._rpe_history])),
            "avg_surprise": float(np.mean([r.surprise_magnitude for r in self._rpe_history])),
            "memories_tracked": len(self._value_estimates),
            "use_learned_values": self.use_learned_values,
            "config": self.get_config(),
        }

        # Add learned value network stats if available
        if self._value_network is not None:
            stats["value_network"] = self._value_network.get_stats()

        return stats

    # ==================== Runtime Configuration Setters ====================

    def set_expected_value(self, memory_id: UUID, value: float) -> None:
        """
        Manually set expected value for a memory.

        Args:
            memory_id: Memory to update
            value: New expected value [0.0, 1.0]
        """
        mem_id_str = str(memory_id)
        value = float(np.clip(value, 0.0, 1.0))
        self._value_estimates[mem_id_str] = value
        logger.info(f"DA expected value for {mem_id_str[:8]}... set to {value:.3f}")

    def set_value_learning_rate(self, rate: float) -> None:
        """
        Set value learning rate (alpha).

        Args:
            rate: Learning rate [0.01, 0.5]
        """
        self.value_learning_rate = float(np.clip(rate, 0.01, 0.5))
        logger.info(f"DA value_learning_rate set to {self.value_learning_rate}")

    def set_default_expected(self, value: float) -> None:
        """
        Set default expected value for new memories.

        Args:
            value: Default expected [0.0, 1.0]
        """
        self.default_expected = float(np.clip(value, 0.0, 1.0))
        logger.info(f"DA default_expected set to {self.default_expected}")

    def set_surprise_threshold(self, threshold: float) -> None:
        """
        Set minimum RPE magnitude to count as surprising.

        Args:
            threshold: Threshold [0.01, 0.2]
        """
        self.surprise_threshold = float(np.clip(threshold, 0.01, 0.2))
        logger.info(f"DA surprise_threshold set to {self.surprise_threshold}")

    def reset_expectations(self) -> None:
        """Clear all learned expectations."""
        self._value_estimates.clear()
        self._outcome_counts.clear()
        logger.info("DA expectations reset")

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "default_expected": self.default_expected,
            "value_learning_rate": self.value_learning_rate,
            "surprise_threshold": self.surprise_threshold,
            "max_rpe_magnitude": self.max_rpe_magnitude,
        }

    def reset_history(self) -> None:
        """Clear RPE history to free memory."""
        self._rpe_history = []

    # MEMORY-HIGH-007 FIX: Add save/load methods for expectation persistence
    def save_state(self) -> dict:
        """
        Save dopamine system state to dictionary.

        Returns:
            Dictionary with value estimates and outcome counts
        """
        state = {
            "value_estimates": self._value_estimates.copy(),
            "outcome_counts": self._outcome_counts.copy(),
            "default_expected": self.default_expected,
            "value_learning_rate": self.value_learning_rate,
            "surprise_threshold": self.surprise_threshold,
            "use_learned_values": self.use_learned_values,
        }

        # Save learned value network if present
        if self._value_network is not None:
            state["value_network"] = self._value_network.save_state()

        return state

    def load_state(self, state: dict) -> None:
        """
        Load dopamine system state from dictionary.

        Args:
            state: Dictionary from save_state()
        """
        self._value_estimates = state.get("value_estimates", {})
        self._outcome_counts = state.get("outcome_counts", {})

        # Optionally update parameters if provided
        if "default_expected" in state:
            self.default_expected = state["default_expected"]
        if "value_learning_rate" in state:
            self.value_learning_rate = state["value_learning_rate"]

        # Load learned value network if present
        if "value_network" in state and self._value_network is not None:
            self._value_network.load_state(state["value_network"])

    def tag_episode_with_rpe(
        self,
        episode: Episode,
        actual_outcome: float,
        embedding: np.ndarray | None = None,
    ) -> Episode:
        """
        Tag an episode with its prediction error for prioritized replay.

        P1-1: Integrates dopamine RPE with consolidation system.
        Episodes with high |PE| are prioritized for replay during sleep.

        Args:
            episode: Episode to tag
            actual_outcome: Observed outcome [0, 1]
            embedding: Episode embedding (optional)

        Returns:
            Episode with prediction_error field set
        """
        rpe_result = self.compute_rpe(
            episode.id,
            actual_outcome,
            embedding=embedding
        )

        # Update episode with prediction error
        episode.prediction_error = abs(rpe_result.rpe)
        episode.prediction_error_timestamp = datetime.now()

        # Also update our expectations for future predictions
        self.update_expectations(
            episode.id,
            actual_outcome,
            embedding=embedding
        )

        logger.debug(
            f"Tagged episode {episode.id} with PE={episode.prediction_error:.3f} "
            f"(expected={rpe_result.expected:.3f}, actual={actual_outcome:.3f})"
        )

        return episode

    async def tag_episodes_batch(
        self,
        episodes_outcomes: list[tuple[Episode, float]],
    ) -> list[Episode]:
        """
        Tag multiple episodes with prediction errors.

        P1-1: Batch processing for efficiency.

        Args:
            episodes_outcomes: List of (episode, outcome) pairs

        Returns:
            List of tagged episodes
        """
        tagged = []
        for episode, outcome in episodes_outcomes:
            tagged.append(self.tag_episode_with_rpe(episode, outcome))
        return tagged


# Convenience function for simple RPE computation
def compute_rpe(
    actual: float,
    expected: float = 0.5
) -> float:
    """
    Simple reward prediction error.

    Args:
        actual: Actual outcome [0, 1]
        expected: Expected outcome [0, 1]

    Returns:
        RPE (δ = actual - expected)
    """
    return actual - expected


__all__ = [
    "DopamineSystem",
    "LearnedValueEstimator",
    "RewardPredictionError",
    "compute_rpe",
]
