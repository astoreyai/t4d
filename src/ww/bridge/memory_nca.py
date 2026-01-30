"""
Memory-NCA Bridge for World Weaver.

Connects WW memory operations to NCA neural field dynamics:

1. ENCODING: Memory embedding modulated by NT state at encoding time
2. RETRIEVAL: Query weighted by current cognitive state
3. LEARNING: Three-factor rule + NCA coupling gradients

Integration Points:
- ww.memory.episodic: EpisodicMemory encoding/retrieval
- ww.learning.dopamine: DopamineSystem RPE
- ww.learning.eligibility: EligibilityTrace
- ww.learning.three_factor: ThreeFactorLearningRule
- ww.nca.*: Neural field, coupling, attractors, energy

Key Insight: Memory operations should be STATE-DEPENDENT.
The same input encoded in FOCUS vs EXPLORE creates different memories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from ww.learning.dopamine import DopamineSystem
    from ww.nca.attractors import CognitiveState, StateTransitionManager
    from ww.nca.coupling import LearnableCoupling
    from ww.nca.energy import EnergyLandscape
    from ww.nca.neural_field import NeuralFieldSolver

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for Memory-NCA bridge."""

    # State-dependent encoding
    encoding_nt_weight: float = 0.3  # How much NT state affects encoding
    state_context_dim: int = 32  # Dimension of state context vector

    # Retrieval modulation
    retrieval_state_matching: bool = True  # Prefer memories from similar states
    state_similarity_weight: float = 0.2  # Weight of state match in retrieval

    # Learning integration
    use_nca_gradients: bool = True  # Include NCA coupling in learning
    coupling_lr_scale: float = 0.5  # Scale NCA learning relative to memory learning

    # Cognitive state effects
    focus_boost: float = 1.3  # Encoding boost in FOCUS state
    explore_diversity: float = 1.5  # Retrieval diversity in EXPLORE
    consolidate_replay: int = 10  # Memories to replay in CONSOLIDATE


@dataclass
class EncodingContext:
    """Context for state-dependent encoding."""
    memory_id: UUID
    embedding: np.ndarray
    nt_state: np.ndarray
    cognitive_state: CognitiveState
    timestamp: datetime = field(default_factory=datetime.now)
    energy: float = 0.0


@dataclass
class RetrievalContext:
    """Context for state-modulated retrieval."""
    query_embedding: np.ndarray
    query_nt_state: np.ndarray
    query_cognitive_state: CognitiveState
    retrieved_ids: list[UUID] = field(default_factory=list)
    state_similarities: list[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryNCABridge:
    """
    Bridge connecting WW memory system to NCA dynamics.

    Responsibilities:
    1. Augment memory embeddings with NT state context
    2. Modulate retrieval based on cognitive state
    3. Route learning signals between systems
    4. Trigger consolidation during CONSOLIDATE state

    Status: 90% complete - missing direct encoding/retrieval hookup.
    See README.md for remaining integration points.
    """

    def __init__(
        self,
        config: BridgeConfig | None = None,
        neural_field: NeuralFieldSolver | None = None,
        coupling: LearnableCoupling | None = None,
        state_manager: StateTransitionManager | None = None,
        energy_landscape: EnergyLandscape | None = None,
        dopamine: DopamineSystem | None = None
    ):
        """
        Initialize memory-NCA bridge.

        Args:
            config: Bridge configuration
            neural_field: NCA neural field solver
            coupling: Learnable coupling matrix
            state_manager: Cognitive state manager
            energy_landscape: Energy-based dynamics
            dopamine: WW dopamine system for RPE
        """
        self.config = config or BridgeConfig()
        self.neural_field = neural_field
        self.coupling = coupling
        self.state_manager = state_manager
        self.energy_landscape = energy_landscape
        self.dopamine = dopamine

        # State-indexed memory cache
        self._encoding_history: list[EncodingContext] = []
        self._retrieval_history: list[RetrievalContext] = []
        self._max_history = 10000

        # State context projection (to be learned)
        self._state_projection = np.random.randn(
            6, self.config.state_context_dim
        ).astype(np.float32) * 0.1

        logger.info("MemoryNCABridge initialized")

    def get_current_nt_state(self) -> np.ndarray:
        """Get current NT state from neural field."""
        if self.neural_field is not None:
            state = self.neural_field.get_mean_state()
            return state.to_array()
        return np.full(6, 0.5, dtype=np.float32)

    def get_current_cognitive_state(self) -> CognitiveState | None:
        """Get current cognitive state."""
        if self.state_manager is not None:
            return self.state_manager.get_current_state()
        return None

    def augment_encoding(
        self,
        embedding: np.ndarray,
        memory_id: UUID
    ) -> tuple[np.ndarray, EncodingContext]:
        """
        Augment memory embedding with NT state context.

        Creates state-dependent encoding by concatenating:
        - Original embedding
        - NT state projection

        Args:
            embedding: Raw memory embedding [dim]
            memory_id: Memory UUID

        Returns:
            (augmented_embedding, encoding_context)
        """
        nt_state = self.get_current_nt_state()
        cognitive_state = self.get_current_cognitive_state()

        # Project NT state to context vector
        state_context = nt_state @ self._state_projection  # [state_context_dim]

        # Apply cognitive state modulation
        if cognitive_state is not None:
            from ww.nca.attractors import CognitiveState

            if cognitive_state == CognitiveState.FOCUS:
                # Boost encoding strength
                state_context *= self.config.focus_boost
            elif cognitive_state == CognitiveState.EXPLORE:
                # Add noise for diversity
                state_context += np.random.randn(
                    self.config.state_context_dim
                ).astype(np.float32) * 0.1

        # Augment embedding
        weight = self.config.encoding_nt_weight
        augmented = np.concatenate([
            embedding * (1 - weight),
            state_context * weight
        ])

        # Compute current energy
        energy = 0.0
        if self.energy_landscape is not None:
            energy = self.energy_landscape.compute_total_energy(nt_state)

        context = EncodingContext(
            memory_id=memory_id,
            embedding=embedding.copy(),
            nt_state=nt_state.copy(),
            cognitive_state=cognitive_state,
            energy=energy
        )

        self._encoding_history.append(context)
        if len(self._encoding_history) > self._max_history:
            self._encoding_history = self._encoding_history[-self._max_history:]

        return augmented, context

    def modulate_retrieval(
        self,
        query_embedding: np.ndarray,
        candidate_contexts: list[EncodingContext],
        top_k: int = 10
    ) -> tuple[list[UUID], RetrievalContext]:
        """
        Modulate retrieval based on state similarity.

        Prefers memories encoded in similar cognitive states.

        Args:
            query_embedding: Query embedding
            candidate_contexts: Encoding contexts of candidates
            top_k: Number to retrieve

        Returns:
            (ranked_memory_ids, retrieval_context)
        """
        current_nt = self.get_current_nt_state()
        current_state = self.get_current_cognitive_state()

        # Compute state similarities
        state_sims = []
        for ctx in candidate_contexts:
            # NT state similarity
            nt_sim = 1 - np.linalg.norm(current_nt - ctx.nt_state) / np.sqrt(6)

            # Cognitive state match bonus
            state_bonus = 0.2 if ctx.cognitive_state == current_state else 0.0

            state_sims.append(nt_sim + state_bonus)

        # Apply state similarity weight
        if self.config.retrieval_state_matching:
            # Combine with embedding similarity (assumed in order)
            # This is a simplified version - full impl would rerank
            weight = self.config.state_similarity_weight
            combined_scores = [(1 - weight) + weight * s for s in state_sims]
        else:
            combined_scores = [1.0] * len(candidate_contexts)

        # Rank by combined score
        ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
        ranked_ids = [candidate_contexts[i].memory_id for i in ranked_indices]
        ranked_sims = [state_sims[i] for i in ranked_indices]

        context = RetrievalContext(
            query_embedding=query_embedding.copy(),
            query_nt_state=current_nt.copy(),
            query_cognitive_state=current_state,
            retrieved_ids=ranked_ids,
            state_similarities=ranked_sims
        )

        self._retrieval_history.append(context)
        if len(self._retrieval_history) > self._max_history:
            self._retrieval_history = self._retrieval_history[-self._max_history:]

        return ranked_ids, context

    def compute_learning_signal(
        self,
        memory_id: UUID,
        outcome: float,
        encoding_context: EncodingContext | None = None
    ) -> dict:
        """
        Compute combined learning signal from WW + NCA.

        Integrates:
        1. Dopamine RPE (from WW)
        2. NCA coupling gradient
        3. Eligibility modulation

        Args:
            memory_id: Memory that was used
            outcome: Actual outcome [0, 1]
            encoding_context: Context from encoding (if available)

        Returns:
            Learning signal dict with components
        """
        signals = {
            "memory_id": str(memory_id),
            "outcome": outcome,
            "rpe": 0.0,
            "coupling_gradient": None,
            "effective_lr": 0.01,
        }

        # Get dopamine RPE
        if self.dopamine is not None:
            rpe_result = self.dopamine.compute_rpe(memory_id, outcome)
            signals["rpe"] = rpe_result.rpe

            # Update dopamine expectations
            self.dopamine.update_expectations(memory_id, outcome)

        # Get NCA coupling gradient
        if self.coupling is not None and self.config.use_nca_gradients:
            current_nt = self.get_current_nt_state()

            # Update coupling based on RPE
            self.coupling.update_from_rpe(
                current_nt,
                signals["rpe"],
                eligibility=None  # Could add eligibility here
            )

            signals["coupling_gradient"] = self.coupling.K.copy()

        # Modulate learning rate by cognitive state
        current_state = self.get_current_cognitive_state()
        if current_state is not None:
            from ww.nca.attractors import CognitiveState

            if current_state == CognitiveState.FOCUS:
                signals["effective_lr"] *= 1.5  # Enhanced learning in focus
            elif current_state == CognitiveState.REST:
                signals["effective_lr"] *= 0.5  # Reduced learning at rest
            elif current_state == CognitiveState.CONSOLIDATE:
                signals["effective_lr"] *= 2.0  # Strong learning during consolidation

        return signals

    def trigger_consolidation(self) -> list[UUID]:
        """
        Trigger memory consolidation (called in CONSOLIDATE state).

        Selects memories for replay based on:
        - Recency
        - RPE magnitude (surprising memories)
        - State diversity

        Returns:
            List of memory IDs to replay
        """
        if not self._encoding_history:
            return []

        # Score memories for replay
        scores = []
        for ctx in self._encoding_history[-1000:]:  # Recent memories
            # Recency score
            age = (datetime.now() - ctx.timestamp).total_seconds()
            recency = np.exp(-age / 3600)  # 1hr decay

            # Energy score (prefer high-energy = unstable memories)
            energy_score = min(ctx.energy / 10, 1.0) if ctx.energy > 0 else 0.5

            scores.append((ctx.memory_id, recency + energy_score))

        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        replay_ids = [s[0] for s in scores[:self.config.consolidate_replay]]

        logger.debug(f"Consolidation triggered: {len(replay_ids)} memories for replay")

        return replay_ids

    def step(self, dt: float = 0.01) -> None:
        """
        Step the bridge forward (called each simulation step).

        Updates:
        - NT state from neural field
        - Cognitive state from state manager
        - Triggers consolidation if in CONSOLIDATE state
        """
        if self.neural_field is not None:
            self.neural_field.step(dt=dt)

        if self.state_manager is not None:
            nt_state = self.get_current_nt_state()
            from ww.nca.neural_field import NeurotransmitterState
            state_obj = NeurotransmitterState.from_array(nt_state)
            self.state_manager.update(state_obj, dt)

            # Check for consolidation
            from ww.nca.attractors import CognitiveState
            if self.state_manager.get_current_state() == CognitiveState.CONSOLIDATE:
                self.trigger_consolidation()

    def get_stats(self) -> dict:
        """Get bridge statistics."""
        return {
            "encoding_history_size": len(self._encoding_history),
            "retrieval_history_size": len(self._retrieval_history),
            "current_nt_state": self.get_current_nt_state().tolist(),
            "current_cognitive_state": (
                self.get_current_cognitive_state().name
                if self.get_current_cognitive_state() else None
            ),
            "config": {
                "encoding_nt_weight": self.config.encoding_nt_weight,
                "retrieval_state_matching": self.config.retrieval_state_matching,
                "use_nca_gradients": self.config.use_nca_gradients,
            }
        }


__all__ = [
    "MemoryNCABridge",
    "BridgeConfig",
    "EncodingContext",
    "RetrievalContext",
]
