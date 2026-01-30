"""
Working Memory Gating with Theta-Gamma Modulation.

Biological Basis:
- Prefrontal cortex maintains working memory via sustained activity
- Alpha oscillations (8-12 Hz) inhibit encoding of distractors
- Gamma oscillations (30-100 Hz) maintain item representations
- Theta phase gates encoding vs. retrieval operations

Mathematical Formulation:

Encoding Gate:
    G_encode = sigmoid(encoding_signal - threshold) * (1 - alpha_inhibition)

Retrieval Gate:
    G_retrieve = sigmoid(retrieval_signal - threshold) * attention_weight

Maintenance via Rehearsal:
    activation(t+1) = activation(t) * decay + G_rehearse * attention
    G_rehearse = (1 - alpha_inhibition) * gamma_amplitude

References:
- Baddeley (2000): Working memory model
- Jensen & Lisman (1998): Theta-gamma coding hypothesis
- Roux & Uhlhaas (2014): Working memory and neural oscillations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WMGatingConfig:
    """Configuration for working memory gating.

    Attributes:
        wm_capacity: Maximum working memory capacity (7±2)
        embed_dim: Embedding dimension
        encoding_threshold: Threshold for encoding gate
        retrieval_threshold: Threshold for retrieval gate
        eviction_threshold: Threshold for evicting items
        decay_rate: Activation decay rate
        rehearsal_boost: Boost from rehearsal
        attention_learning_rate: Learning rate for attention
        alpha_inhibition_weight: Weight of alpha inhibition
        gamma_maintenance_weight: Weight of gamma for maintenance
        theta_frequency: Theta oscillation frequency (Hz)
        gamma_frequency: Gamma oscillation frequency (Hz)
    """
    wm_capacity: int = 7
    dynamic_capacity: bool = False  # If True, capacity adjusts based on NE
    embed_dim: int = 1024
    encoding_threshold: float = 0.5
    retrieval_threshold: float = 0.5
    eviction_threshold: float = 0.2
    decay_rate: float = 0.1
    rehearsal_boost: float = 0.3
    attention_learning_rate: float = 0.1
    alpha_inhibition_weight: float = 0.5
    gamma_maintenance_weight: float = 0.3
    theta_frequency: float = 6.0
    gamma_frequency: float = 40.0


@dataclass
class WMItem:
    """Item stored in working memory.

    Attributes:
        embedding: Item embedding
        activation: Current activation level
        attention: Attention weight
        age: Time steps since encoding
        gamma_phase: Current gamma phase
        priority: Computed priority
    """
    embedding: np.ndarray
    activation: float = 1.0
    attention: float = 1.0
    age: int = 0
    gamma_phase: float = 0.0
    priority: float = 1.0


class EncodingGate:
    """
    Gate controlling what enters working memory.

    Encoding is modulated by:
    1. Encoding signal strength (relevance)
    2. Alpha inhibition (filtering distractors)
    3. Available capacity
    """

    def __init__(
        self,
        threshold: float = 0.5,
        alpha_weight: float = 0.5,
    ):
        """
        Initialize encoding gate.

        Args:
            threshold: Encoding threshold
            alpha_weight: Weight of alpha inhibition
        """
        self.threshold = threshold
        self.alpha_weight = alpha_weight

        logger.debug(f"EncodingGate: threshold={threshold}")

    def compute_gate(
        self,
        encoding_signal: float,
        alpha: float,
        capacity: int,
        max_capacity: int,
    ) -> float:
        """
        Compute encoding gate value.

        G_encode = sigmoid(signal - threshold) * (1 - alpha_inhibition) * capacity_factor

        Args:
            encoding_signal: Strength of encoding signal
            alpha: Alpha oscillation amplitude (0-1)
            capacity: Current items in WM
            max_capacity: Maximum WM capacity

        Returns:
            Gate value [0, 1]
        """
        # Base gate from encoding signal
        gate = self._sigmoid(encoding_signal - self.threshold)

        # Alpha inhibition
        alpha_inhibition = self.alpha_weight * alpha
        gate = gate * (1 - alpha_inhibition)

        # Capacity factor (harder to encode when full)
        if capacity >= max_capacity:
            gate = gate * 0.1  # Strong suppression when full
        else:
            capacity_factor = 1 - (capacity / max_capacity) ** 2
            gate = gate * capacity_factor

        return float(np.clip(gate, 0, 1))

    def should_encode(
        self,
        item_priority: float,
        gate_value: float,
    ) -> bool:
        """
        Decide whether to encode an item.

        Args:
            item_priority: Priority of the item
            gate_value: Current gate value

        Returns:
            Whether to encode
        """
        return (item_priority * gate_value) > self.threshold

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))


class RetrievalGate:
    """
    Gate controlling retrieval from working memory.

    Retrieval is modulated by:
    1. Retrieval signal strength (query match)
    2. Item attention weight
    3. Item activation level
    """

    def __init__(
        self,
        threshold: float = 0.5,
    ):
        """
        Initialize retrieval gate.

        Args:
            threshold: Retrieval threshold
        """
        self.threshold = threshold

        logger.debug(f"RetrievalGate: threshold={threshold}")

    def compute_gate(
        self,
        retrieval_signal: float,
        attention: float,
        activation: float,
    ) -> float:
        """
        Compute retrieval gate value.

        G_retrieve = sigmoid(signal - threshold) * attention * activation

        Args:
            retrieval_signal: Strength of retrieval cue
            attention: Item's attention weight
            activation: Item's activation level

        Returns:
            Gate value [0, 1]
        """
        gate = self._sigmoid(retrieval_signal - self.threshold)
        gate = gate * attention * activation

        return float(np.clip(gate, 0, 1))

    def retrieve_strength(
        self,
        gate_value: float,
        activation: float,
    ) -> float:
        """
        Compute retrieval strength.

        Args:
            gate_value: Current gate value
            activation: Item activation

        Returns:
            Retrieval strength
        """
        return gate_value * activation

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))


class MaintenanceController:
    """
    Controls maintenance of items in working memory.

    Maintenance mechanisms:
    1. Activation decay over time
    2. Rehearsal boosts activation
    3. Gamma oscillations sustain representations
    4. Attention modulates maintenance
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        rehearsal_boost: float = 0.3,
        gamma_weight: float = 0.3,
    ):
        """
        Initialize maintenance controller.

        Args:
            decay_rate: Rate of activation decay
            rehearsal_boost: Boost from rehearsal
            gamma_weight: Weight of gamma for maintenance
        """
        self.decay_rate = decay_rate
        self.rehearsal_boost = rehearsal_boost
        self.gamma_weight = gamma_weight

        logger.debug(
            f"MaintenanceController: decay={decay_rate}, "
            f"rehearsal={rehearsal_boost}"
        )

    def update_activations(
        self,
        activations: np.ndarray,
        attention: np.ndarray,
        gamma: float,
        alpha: float,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Update activations for all items.

        activation(t+1) = activation(t) * decay + maintenance_signal

        Args:
            activations: Current activations [n_items]
            attention: Attention weights [n_items]
            gamma: Gamma amplitude
            alpha: Alpha amplitude (inhibition)
            dt: Time step

        Returns:
            Updated activations [n_items]
        """
        # Decay
        decay_factor = 1 - self.decay_rate * dt
        new_activations = activations * decay_factor

        # Maintenance signal from gamma
        gamma_maintenance = self.gamma_weight * gamma * (1 - alpha)

        # Attention-weighted maintenance
        maintenance = attention * gamma_maintenance

        # Update
        new_activations = new_activations + maintenance * dt

        # Clamp
        new_activations = np.clip(new_activations, 0, 1)

        return new_activations

    def select_for_rehearsal(
        self,
        items: list[WMItem],
        attention: np.ndarray,
        gamma_phase: float,
    ) -> int:
        """
        Select item for rehearsal based on theta phase.

        Args:
            items: List of WM items
            attention: Attention weights
            gamma_phase: Current gamma phase

        Returns:
            Index of item to rehearse (-1 if none)
        """
        if not items:
            return -1

        # Select based on attention and activation
        priorities = []
        for i, item in enumerate(items):
            # Combine attention, activation, and phase
            phase_factor = 0.5 + 0.5 * np.cos(gamma_phase - item.gamma_phase)
            priority = attention[i] * item.activation * phase_factor
            priorities.append(priority)

        # Select highest priority
        return int(np.argmax(priorities))


class WorkingMemoryGating:
    """
    Complete working memory system with oscillatory gating.

    Integrates:
    1. Encoding gate (what enters WM)
    2. Retrieval gate (what is retrieved)
    3. Maintenance controller (what is maintained)
    4. Theta-gamma synchronization

    The system operates in theta cycles, with gamma slots
    for individual items (capacity ~7).
    """

    def __init__(self, config: WMGatingConfig | None = None):
        """
        Initialize working memory gating system.

        Args:
            config: WM configuration
        """
        self.config = config or WMGatingConfig()

        # Gates
        self.encoding_gate = EncodingGate(
            threshold=self.config.encoding_threshold,
            alpha_weight=self.config.alpha_inhibition_weight,
        )
        self.retrieval_gate = RetrievalGate(
            threshold=self.config.retrieval_threshold,
        )
        self.maintenance = MaintenanceController(
            decay_rate=self.config.decay_rate,
            rehearsal_boost=self.config.rehearsal_boost,
            gamma_weight=self.config.gamma_maintenance_weight,
        )

        # Working memory store
        self.items: list[WMItem] = []
        self.attention_weights = np.array([], dtype=np.float32)
        self._ne_level = 0.6  # Track NE for dynamic capacity

        # Oscillation state
        self.theta_phase = 0.0
        self.gamma_phase = 0.0
        self.alpha_amplitude = 0.0

        # Time tracking
        self._time = 0.0
        self._encoding_count = 0
        self._retrieval_count = 0

        logger.info(
            f"WorkingMemoryGating initialized: capacity={self.config.wm_capacity}"
        )

    def step(
        self,
        new_items: list[np.ndarray] | None = None,
        query: np.ndarray | None = None,
        dt_ms: float = 1.0,
    ) -> dict:
        """
        Single timestep of WM operation.

        Args:
            new_items: New items to potentially encode
            query: Query for retrieval
            dt_ms: Time step in milliseconds

        Returns:
            Dict with step results
        """
        result = {
            "encoded": [],
            "retrieved": None,
            "evicted": [],
            "maintained": len(self.items),
        }

        # Update oscillations
        self._update_oscillations(dt_ms)

        # Try encoding new items
        if new_items:
            for emb in new_items:
                encoded = self._try_encode(emb)
                if encoded:
                    result["encoded"].append(len(self.items) - 1)

        # Try retrieval
        if query is not None:
            retrieved = self._try_retrieve(query)
            result["retrieved"] = retrieved

        # Maintenance update
        self._maintenance_step(dt_ms)

        # Eviction check
        evicted = self._eviction_check()
        result["evicted"] = evicted

        # Age items
        for item in self.items:
            item.age += 1

        self._time += dt_ms
        result["time"] = self._time
        result["theta_phase"] = self.theta_phase
        result["gamma_phase"] = self.gamma_phase

        return result

    def _try_encode(self, embedding: np.ndarray) -> bool:
        """Try to encode item into WM."""
        # Compute encoding signal (priority)
        priority = self.compute_attention_modulated_priority(
            embedding, self._get_context_embedding()
        )

        # ATOM-P3-36: Dynamic capacity based on NE
        max_capacity = self.config.wm_capacity
        if self.config.dynamic_capacity:
            # Gaussian curve centered at NE=0.6
            capacity_scale = np.exp(-((self._ne_level - 0.6) ** 2) / (2 * 0.2 ** 2))
            max_capacity = int(self.config.wm_capacity * capacity_scale)

        # Compute gate
        gate = self.encoding_gate.compute_gate(
            encoding_signal=priority,
            alpha=self.alpha_amplitude,
            capacity=len(self.items),
            max_capacity=max_capacity,
        )

        # Decide
        if self.encoding_gate.should_encode(priority, gate):
            # Create item
            item = WMItem(
                embedding=embedding,
                activation=gate,
                attention=priority,
                age=0,
                gamma_phase=self.gamma_phase,
                priority=priority,
            )

            # Add to WM
            self.items.append(item)
            self.attention_weights = np.append(
                self.attention_weights, priority
            )

            # Evict if over capacity
            while len(self.items) > self.config.wm_capacity:
                self._evict_lowest()

            self._encoding_count += 1
            return True

        return False

    def _try_retrieve(
        self,
        query: np.ndarray,
    ) -> tuple[int, np.ndarray, float] | None:
        """Try to retrieve item matching query."""
        if not self.items:
            return None

        best_idx = -1
        best_strength = 0.0

        for i, item in enumerate(self.items):
            # Compute retrieval signal (similarity)
            similarity = np.dot(query, item.embedding) / (
                np.linalg.norm(query) * np.linalg.norm(item.embedding) + 1e-8
            )

            # Compute gate
            gate = self.retrieval_gate.compute_gate(
                retrieval_signal=float(similarity),
                attention=item.attention,
                activation=item.activation,
            )

            strength = self.retrieval_gate.retrieve_strength(
                gate, item.activation
            )

            if strength > best_strength:
                best_strength = strength
                best_idx = i

        if best_idx >= 0 and best_strength > self.config.retrieval_threshold:
            item = self.items[best_idx]

            # Boost activation on retrieval
            item.activation = min(1.0, item.activation + 0.2)
            item.attention = min(1.0, item.attention + 0.1)

            self._retrieval_count += 1
            return (best_idx, item.embedding, best_strength)

        return None

    def _maintenance_step(self, dt_ms: float) -> None:
        """Update maintenance for all items."""
        if not self.items:
            return

        # Get current activations
        activations = np.array([item.activation for item in self.items])

        # Compute gamma amplitude
        gamma = 0.5 + 0.5 * np.sin(self.gamma_phase)

        # Update activations
        new_activations = self.maintenance.update_activations(
            activations=activations,
            attention=self.attention_weights,
            gamma=gamma,
            alpha=self.alpha_amplitude,
            dt=dt_ms / 1000.0,
        )

        # Store back
        for i, item in enumerate(self.items):
            item.activation = float(new_activations[i])

        # Select item for rehearsal
        rehearsal_idx = self.maintenance.select_for_rehearsal(
            self.items,
            self.attention_weights,
            self.gamma_phase,
        )

        if rehearsal_idx >= 0:
            # Boost rehearsed item
            self.items[rehearsal_idx].activation = min(
                1.0,
                self.items[rehearsal_idx].activation +
                self.config.rehearsal_boost * (dt_ms / 1000.0)
            )

    def _eviction_check(self) -> list[int]:
        """Check and evict low-activation items."""
        evicted = []

        # Find items below threshold
        to_remove = []
        for i, item in enumerate(self.items):
            if item.activation < self.config.eviction_threshold:
                to_remove.append(i)

        # Remove (reverse order to preserve indices)
        for i in reversed(to_remove):
            self.items.pop(i)
            self.attention_weights = np.delete(self.attention_weights, i)
            evicted.append(i)

        return evicted

    def _evict_lowest(self) -> None:
        """Evict lowest priority item."""
        if not self.items:
            return

        # Find lowest activation * attention
        priorities = [
            item.activation * item.attention
            for item in self.items
        ]
        min_idx = int(np.argmin(priorities))

        self.items.pop(min_idx)
        self.attention_weights = np.delete(self.attention_weights, min_idx)

    def _update_oscillations(self, dt_ms: float) -> None:
        """Update oscillation phases."""
        dt_s = dt_ms / 1000.0

        # Update theta
        self.theta_phase += 2 * np.pi * self.config.theta_frequency * dt_s
        self.theta_phase = self.theta_phase % (2 * np.pi)

        # Update gamma (faster)
        self.gamma_phase += 2 * np.pi * self.config.gamma_frequency * dt_s
        self.gamma_phase = self.gamma_phase % (2 * np.pi)

        # Alpha tracks inverse of theta (simplified)
        self.alpha_amplitude = 0.3 + 0.2 * np.cos(self.theta_phase)

    def _get_context_embedding(self) -> np.ndarray:
        """Get context embedding from current WM contents."""
        if not self.items:
            return np.zeros(self.config.embed_dim, dtype=np.float32)

        # Attention-weighted average
        embeddings = np.stack([item.embedding for item in self.items])
        weights = self.attention_weights / (
            np.sum(self.attention_weights) + 1e-8
        )
        context = np.sum(embeddings * weights[:, np.newaxis], axis=0)

        return context

    def compute_attention_modulated_priority(
        self,
        item_emb: np.ndarray,
        context_emb: np.ndarray,
    ) -> float:
        """
        Compute priority for an item given context.

        Args:
            item_emb: Item embedding
            context_emb: Context embedding

        Returns:
            Priority value
        """
        # Similarity to context
        if np.linalg.norm(context_emb) < 1e-8:
            return 0.5

        similarity = np.dot(item_emb, context_emb) / (
            np.linalg.norm(item_emb) * np.linalg.norm(context_emb) + 1e-8
        )

        # Transform to priority
        priority = 0.5 + 0.5 * float(similarity)

        return priority

    def synchronize_with_theta_gamma(self) -> None:
        """Synchronize item gamma phases with theta."""
        if not self.items:
            return

        # Assign gamma phases based on position
        for i, item in enumerate(self.items):
            slot = i % self.config.wm_capacity
            item.gamma_phase = (
                2 * np.pi * slot / self.config.wm_capacity
            )

    def get_contents(self) -> list[tuple[np.ndarray, float, float]]:
        """Get current WM contents with metadata."""
        return [
            (item.embedding, item.activation, item.attention)
            for item in self.items
        ]

    def get_stats(self) -> dict:
        """Get WM statistics."""
        return {
            "capacity": self.config.wm_capacity,
            "current_size": len(self.items),
            "encoding_count": self._encoding_count,
            "retrieval_count": self._retrieval_count,
            "mean_activation": (
                float(np.mean([item.activation for item in self.items]))
                if self.items else 0.0
            ),
            "mean_attention": (
                float(np.mean(self.attention_weights))
                if len(self.attention_weights) > 0 else 0.0
            ),
            "theta_phase": self.theta_phase,
            "gamma_phase": self.gamma_phase,
            "alpha_amplitude": self.alpha_amplitude,
        }


__all__ = [
    "WMGatingConfig",
    "WMItem",
    "EncodingGate",
    "RetrievalGate",
    "MaintenanceController",
    "WorkingMemoryGating",
]
