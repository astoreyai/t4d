"""
Consolidation-Plasticity Integration for T4DM.

Bridges the temporal dynamics, embedding modulation, and memory systems.
Implements the recommendations from neurocomputational analysis:

1. NeuromodulatorStateAdapter: Bridge type mismatch between systems
2. LearnedSalienceProvider: Extract salience from learned gates
3. ConsolidationAwareModulation: Apply consolidation-appropriate states
4. PlasticityCoordinator: Orchestrate reconsolidation with modulation

Hinton-inspired: Create information flow from what the system learns
matters (plasticity) back to how embeddings are modulated (representation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID, uuid4

import numpy as np

from t4dm.embedding.modulated import (
    ModulatedEmbeddingAdapter,
    NeuromodulatorState,
)
from t4dm.learning.homeostatic import HomeostaticPlasticity
from t4dm.learning.reconsolidation import ReconsolidationEngine, ReconsolidationUpdate

if TYPE_CHECKING:
    from t4dm.learning.neuromodulators import NeuromodulatorState as OrchestraState

logger = logging.getLogger(__name__)


# ============================================================================
# State Adapter - Bridge between NeuromodulatorOrchestra and ModulatedAdapter
# ============================================================================


def adapt_orchestra_state(orchestra_state: OrchestraState) -> NeuromodulatorState:
    """
    Convert NeuromodulatorOrchestra state to embedding modulation state.

    The orchestra produces:
    - dopamine_rpe: float (reward prediction error, centered at 0)
    - norepinephrine_gain: float (arousal gain, typically 0-2)
    - acetylcholine_mode: str ("encoding", "balanced", "retrieval")
    - serotonin_mood: float (0-1)
    - inhibition_sparsity: float (0-1)

    The modulation adapter expects:
    - acetylcholine: float (0-1, high = encoding)
    - dopamine: float (0-1, salience signal)
    - norepinephrine: float (0-1, exploration)
    - serotonin: float (0-1, long-term perspective)

    Args:
        orchestra_state: State from NeuromodulatorOrchestra

    Returns:
        NeuromodulatorState for embedding modulation
    """
    # Convert ACh mode to continuous level
    ach_level = {
        "encoding": 0.9,
        "balanced": 0.5,
        "retrieval": 0.2,
    }.get(orchestra_state.acetylcholine_mode, 0.5)

    # Convert RPE to 0-1 dopamine level (center at 0.5)
    dopamine = 0.5 + np.clip(orchestra_state.dopamine_rpe, -0.5, 0.5)

    # Convert gain to 0-1 norepinephrine (assuming gain is typically 0-2)
    norepinephrine = np.clip(orchestra_state.norepinephrine_gain / 2.0, 0, 1)

    return NeuromodulatorState(
        acetylcholine=float(ach_level),
        dopamine=float(dopamine),
        norepinephrine=float(norepinephrine),
        serotonin=float(orchestra_state.serotonin_mood),
    )


# ============================================================================
# Salience Provider - Extract learned importance from gate weights
# ============================================================================


class SalienceProvider(Protocol):
    """Protocol for providing learned salience weights."""

    def get_salience_weights(self, dimension: int) -> np.ndarray:
        """Get salience weights for embedding dimensions."""
        ...


@dataclass
class LearnedSalienceProvider:
    """
    Extract salience weights from learned gate projections.

    The LearnedMemoryGate learns W_content (content -> gate) during training.
    Dimensions with high weight magnitude are more important for utility.

    This creates information flow: what the system learns matters ->
    how embeddings are modulated.
    """

    gate_weights: np.ndarray | None = None
    fallback_weights: np.ndarray | None = None

    def set_gate_weights(self, weights: np.ndarray) -> None:
        """
        Set gate weights from LearnedMemoryGate.

        Args:
            weights: W_content matrix from gate (output_dim x input_dim)
        """
        # Sum of squared weights per input dimension indicates importance
        importance = np.sum(weights ** 2, axis=0)
        importance = importance / (importance.sum() + 1e-8)
        self.gate_weights = importance.astype(np.float32)

    def get_salience_weights(self, dimension: int) -> np.ndarray:
        """
        Get salience weights for embedding dimensions.

        Args:
            dimension: Expected embedding dimension

        Returns:
            Normalized salience weights
        """
        if self.gate_weights is not None and len(self.gate_weights) == dimension:
            return self.gate_weights

        if self.fallback_weights is not None and len(self.fallback_weights) == dimension:
            return self.fallback_weights

        # Default: uniform weights
        return np.ones(dimension, dtype=np.float32) / dimension


# ============================================================================
# Consolidation-Aware Modulation States
# ============================================================================


def get_consolidation_state() -> NeuromodulatorState:
    """
    Get neuromodulator state appropriate for consolidation.

    During consolidation:
    - Low ACh: Not actively encoding or retrieving
    - High DA: Strengthen salient/important patterns
    - Low NE: No exploration, focused processing
    - High 5-HT: Long-term perspective, patience
    """
    return NeuromodulatorState(
        acetylcholine=0.1,
        dopamine=0.7,
        norepinephrine=0.2,
        serotonin=0.8,
    )


def get_sleep_replay_state() -> NeuromodulatorState:
    """
    Get neuromodulator state for sleep replay.

    During replay (sharp-wave ripples):
    - Very low ACh: Hippocampal output mode
    - Moderate DA: Pattern reactivation
    - Very low NE: REM atonia equivalent
    - High 5-HT: Integration across time
    """
    return NeuromodulatorState(
        acetylcholine=0.05,
        dopamine=0.6,
        norepinephrine=0.1,
        serotonin=0.9,
    )


def get_pattern_separation_state() -> NeuromodulatorState:
    """
    Get neuromodulator state for pattern separation.

    During pattern separation:
    - High ACh: Maximum encoding discrimination
    - Low DA: Don't amplify similarities
    - Moderate NE: Some noise for differentiation
    - Low 5-HT: Focus on immediate patterns
    """
    return NeuromodulatorState(
        acetylcholine=0.95,
        dopamine=0.3,
        norepinephrine=0.5,
        serotonin=0.3,
    )


# ============================================================================
# Plasticity Coordinator
# ============================================================================


@dataclass
class PlasticityConfig:
    """Configuration for plasticity coordination."""

    # Reconsolidation settings
    max_update_per_outcome: int = 10
    cooldown_seconds: float = 60.0

    # Homeostatic settings
    target_norm: float = 1.0
    homeostatic_interval: int = 100  # Updates between homeostatic passes

    # Modulation integration
    apply_modulation_to_updates: bool = True
    modulation_strength: float = 0.5  # Blend between raw and modulated


class PlasticityCoordinator:
    """
    Coordinate plasticity mechanisms with embedding modulation.

    Orchestrates:
    1. Reconsolidation updates based on outcomes
    2. Homeostatic scaling to maintain stability
    3. Modulation-aware updates (use appropriate neuromodulator state)

    The key insight: updates should be modulated by the same state
    that generated the retrieval, creating consistent learning.
    """

    def __init__(
        self,
        config: PlasticityConfig | None = None,
        modulated_adapter: ModulatedEmbeddingAdapter | None = None,
    ):
        """Initialize plasticity coordinator."""
        self._config = config or PlasticityConfig()
        self._adapter = modulated_adapter

        # Sub-systems
        self._reconsolidation = ReconsolidationEngine(
            max_update_magnitude=0.1,
            cooldown_hours=self._config.cooldown_seconds / 3600.0,
        )
        self._homeostatic = HomeostaticPlasticity(
            target_norm=self._config.target_norm,
        )

        # Salience provider
        self._salience_provider: SalienceProvider | None = None

        # Tracking
        self._update_count = 0
        self._last_homeostatic = datetime.now()

    def set_salience_provider(self, provider: SalienceProvider) -> None:
        """Set the salience weight provider."""
        self._salience_provider = provider

    def set_modulated_adapter(self, adapter: ModulatedEmbeddingAdapter) -> None:
        """Set the modulated embedding adapter."""
        self._adapter = adapter

    async def process_outcome(
        self,
        outcome_score: float,
        retrieved_embeddings: list[np.ndarray],
        query_embedding: np.ndarray,
        memory_ids: list[str],
        current_state: NeuromodulatorState | None = None,
    ) -> list[ReconsolidationUpdate]:
        """
        Process an outcome and generate reconsolidation updates.

        Args:
            outcome_score: Outcome score (0-1)
            retrieved_embeddings: Embeddings of retrieved memories
            query_embedding: The query that triggered retrieval
            memory_ids: IDs of the memories
            current_state: Neuromodulator state during retrieval

        Returns:
            List of reconsolidation updates to apply
        """
        updates = []

        # Limit updates per outcome
        n_updates = min(len(retrieved_embeddings), self._config.max_update_per_outcome)

        for i in range(n_updates):
            original_emb = retrieved_embeddings[i]
            memory_id = memory_ids[i]

            # Compute update using reconsolidate method
            updated_emb = self._reconsolidation.reconsolidate(
                memory_id=uuid4() if not isinstance(memory_id, UUID) else memory_id,
                memory_embedding=original_emb,
                query_embedding=query_embedding,
                outcome_score=outcome_score,
            )

            if updated_emb is not None:
                # Apply modulation if configured
                if self._config.apply_modulation_to_updates and self._adapter:
                    # Use retrieval state for consistency
                    state = current_state or NeuromodulatorState.for_retrieval()
                    self._adapter.set_state(state)

                    # Blend modulated and raw update
                    modulated = self._adapter._apply_modulation(updated_emb)
                    updated_emb = (
                        (1 - self._config.modulation_strength) * updated_emb +
                        self._config.modulation_strength * modulated
                    )

                    # Re-normalize
                    norm = np.linalg.norm(updated_emb)
                    if norm > 1e-8:
                        updated_emb = updated_emb / norm

                update = ReconsolidationUpdate(
                    memory_id=memory_id,
                    query_embedding=query_embedding,
                    original_embedding=original_emb,
                    updated_embedding=updated_emb,
                    outcome_score=outcome_score,
                    advantage=outcome_score - 0.5,
                    learning_rate=0.01,
                )
                updates.append(update)

        self._update_count += len(updates)

        # Periodic homeostatic pass
        if self._update_count % self._config.homeostatic_interval == 0:
            self._run_homeostatic_pass(updates)

        return updates

    def _run_homeostatic_pass(self, recent_updates: list[ReconsolidationUpdate]) -> None:
        """Run homeostatic scaling on recent updates."""
        for update in recent_updates:
            self._homeostatic.update_statistics(update.updated_embedding)

    def get_stats(self) -> dict:
        """Get coordinator statistics."""
        return {
            "update_count": self._update_count,
            "homeostatic_state": self._homeostatic.get_state().to_dict(),
            "has_salience_provider": self._salience_provider is not None,
            "has_modulated_adapter": self._adapter is not None,
        }


# ============================================================================
# Integration Factory
# ============================================================================


def create_plasticity_coordinator(
    modulated_adapter: ModulatedEmbeddingAdapter | None = None,
    config: PlasticityConfig | None = None,
) -> PlasticityCoordinator:
    """Create a configured plasticity coordinator."""
    return PlasticityCoordinator(
        config=config,
        modulated_adapter=modulated_adapter,
    )


__all__ = [
    # State adaptation
    "adapt_orchestra_state",
    # Salience
    "SalienceProvider",
    "LearnedSalienceProvider",
    # Consolidation states
    "get_consolidation_state",
    "get_sleep_replay_state",
    "get_pattern_separation_state",
    # Plasticity coordination
    "PlasticityConfig",
    "PlasticityCoordinator",
    "create_plasticity_coordinator",
]
