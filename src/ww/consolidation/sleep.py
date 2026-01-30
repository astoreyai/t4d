"""
Sleep-Based Consolidation for World Weaver.

Addresses Hinton critique: Consolidation is manual/scheduled, not replay-based.
Implements biologically-inspired sleep phases for memory consolidation.

Biological Basis:
- NREM (Slow-Wave Sleep): Sharp-wave ripples replay recent experiences
  - High-value experiences are prioritized for replay
  - Hippocampal → neocortical transfer strengthens semantic memories
  - ~75% of sleep time, critical for declarative memory

- REM Sleep: Creative integration and abstraction
  - Dreams combine disparate memories
  - Pattern finding across clusters creates abstract concepts
  - ~25% of sleep time, critical for procedural memory

- Synaptic Downscaling: Homeostatic pruning during sleep
  - Weak connections are pruned
  - Strong connections are preserved (synaptic tagging)
  - Maintains overall network stability

- Sharp-Wave Ripples (SWR): Fast compressed replay events
  - ~100ms high-frequency bursts in hippocampus
  - Compress temporal sequences for efficient transfer
  - Critical for hippocampal-cortical dialogue

References:
- Wilson & McNaughton (1994): Reactivation of hippocampal ensemble memories during sleep
- Foster & Wilson (2006): Reverse replay of behavioural sequences in hippocampal place cells
- McClelland, McNaughton & O'Reilly (1995): Why there are complementary learning systems
- Diekelmann & Born (2010): The memory function of sleep
- Stickgold (2005): Sleep-dependent memory consolidation

Implementation:
1. NREM Phase: Replay high-value episodes with SWR compression
2. REM Phase: Cluster analysis to find abstract patterns
3. Prune Phase: Synaptic downscaling and weak connection removal
4. Full Sleep Cycle: Alternating NREM-REM phases with final pruning
5. Homeostatic Integration: Apply synaptic scaling during consolidation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol
from uuid import UUID

import numpy as np

from ww.learning.generative_replay import GenerativeReplayConfig, GenerativeReplaySystem
from ww.learning.vae_generator import VAEConfig, VAEGenerator
from ww.learning.vae_training import VAEReplayTrainer, VAETrainingConfig

logger = logging.getLogger(__name__)


class SleepPhase(Enum):
    """Sleep phases."""
    NREM = "nrem"
    REM = "rem"
    PRUNE = "prune"
    WAKE = "wake"


class ReplayDirection(Enum):
    """Direction of memory replay during SWR.

    Biological Basis (Foster & Wilson 2006):
    - FORWARD: Sequences play in experienced order. Occurs during theta states,
      used for planning and prediction. ~10% of rest replays.
    - REVERSE: Sequences play backwards. Occurs during rest/sleep SWRs,
      critical for credit assignment (reward propagation). ~90% of rest replays.
    - BIDIRECTIONAL: Both directions interleaved. Some SWR events show this.

    Functional Roles:
    - Forward: Predicts upcoming states, planning
    - Reverse: Propagates TD error/reward backwards for credit assignment
    """
    FORWARD = "forward"      # Replay in experienced order (planning)
    REVERSE = "reverse"      # Replay backwards (credit assignment)
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class SleepCycleResult:
    """Result of a complete sleep cycle."""

    session_id: str
    start_time: datetime
    end_time: datetime
    nrem_replays: int
    rem_abstractions: int
    pruned_connections: int
    strengthened_connections: int
    total_duration_seconds: float

    @property
    def duration(self) -> timedelta:
        """Total duration of sleep cycle."""
        return self.end_time - self.start_time


@dataclass
class ReplayEvent:
    """Record of a memory replay during NREM."""

    episode_id: UUID
    replay_time: datetime = field(default_factory=datetime.now)
    priority_score: float = 0.0
    strengthening_applied: bool = False
    entities_extracted: list[str] = field(default_factory=list)
    direction: ReplayDirection = ReplayDirection.REVERSE  # Default: reverse (credit assignment)
    sequence_position: int = 0  # Position in replay sequence
    rpe: float = 0.0  # P1D: RPE generated during replay


@dataclass
class AbstractionEvent:
    """Record of concept abstraction during REM.

    Represents an abstract concept extracted from clustered entities
    during REM sleep. The concept is persisted as a semantic Entity
    with relationships to source entities.

    Attributes:
        cluster_ids: Node IDs of source entities in cluster
        abstraction_time: When abstraction occurred
        concept_name: Generated name for the concept
        confidence: Confidence score from clustering similarity
        entity_id: UUID of stored concept Entity (if persisted)
        centroid_embedding: Computed centroid of cluster (for retrieval)
    """

    cluster_ids: list[str]
    abstraction_time: datetime = field(default_factory=datetime.now)
    concept_name: str | None = None
    confidence: float = 0.0
    entity_id: UUID | None = None
    centroid_embedding: list[float] | None = None


class SharpWaveRipple:
    """
    Sharp-wave ripple (SWR) generator for compressed memory replay.

    SWRs are brief (~100ms) high-frequency oscillations in hippocampus
    that compress and replay recent experiences at ~10-20x speed.
    They're critical for hippocampal-cortical memory transfer.

    Replay Direction (Foster & Wilson 2006):
    - REVERSE (~90% during rest): Backwards replay propagates TD error/reward
      for credit assignment. Most common during NREM sleep SWRs.
    - FORWARD (~10% during rest): Forward replay for planning and prediction.
      More common during theta states and awake replay.

    Implementation:
    - Select sequences of related memories
    - Compress temporal structure
    - Apply direction (forward/reverse/bidirectional)
    - Generate rapid replay events for consolidation
    """

    # Biological proportions (Foster & Wilson 2006)
    REVERSE_PROBABILITY = 0.9  # 90% reverse during rest
    FORWARD_PROBABILITY = 0.1  # 10% forward during rest

    def __init__(
        self,
        compression_factor: float = 10.0,
        min_sequence_length: int = 3,
        max_sequence_length: int = 8,
        coherence_threshold: float = 0.5,
        default_direction: ReplayDirection = ReplayDirection.REVERSE
    ):
        """
        Initialize SWR generator.

        Args:
            compression_factor: Temporal compression ratio
            min_sequence_length: Minimum memories per ripple
            max_sequence_length: Maximum memories per ripple
            coherence_threshold: Minimum similarity for sequence
            default_direction: Default replay direction (REVERSE for credit assignment)
        """
        self.compression_factor = compression_factor
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.coherence_threshold = coherence_threshold
        self.default_direction = default_direction

        self._ripple_count = 0
        self._total_memories_replayed = 0
        self._forward_replays = 0
        self._reverse_replays = 0

    def generate_ripple_sequence(
        self,
        episodes: list[Any],
        seed_idx: int | None = None,
        direction: ReplayDirection | None = None
    ) -> tuple[list[Any], ReplayDirection]:
        """
        Generate a sharp-wave ripple sequence from episodes.

        Selects a coherent sequence of related memories for
        compressed replay during NREM sleep.

        Args:
            episodes: Available episodes with embeddings
            seed_idx: Optional seed index (random if None)
            direction: Replay direction. If None, uses probabilistic selection
                       (90% reverse, 10% forward per Foster & Wilson 2006)

        Returns:
            Tuple of (sequence of episodes, direction used)
        """
        import random

        if len(episodes) < self.min_sequence_length:
            return [], self.default_direction

        # Determine direction
        if direction is None:
            # Probabilistic selection per biological proportions
            if random.random() < self.REVERSE_PROBABILITY:
                direction = ReplayDirection.REVERSE
            else:
                direction = ReplayDirection.FORWARD

        # Select seed
        if seed_idx is None:
            seed_idx = random.randint(0, len(episodes) - 1)

        seed = episodes[seed_idx]
        seed_emb = self._get_embedding(seed)
        if seed_emb is None:
            return [], direction

        # Build coherent sequence (always forward first, then apply direction)
        sequence = [seed]
        used = {seed_idx}

        while len(sequence) < self.max_sequence_length:
            best_idx = -1
            best_sim = -1.0

            last_emb = self._get_embedding(sequence[-1])
            if last_emb is None:
                break

            for i, ep in enumerate(episodes):
                if i in used:
                    continue

                ep_emb = self._get_embedding(ep)
                if ep_emb is None:
                    continue

                sim = self._cosine_similarity(last_emb, ep_emb)
                if sim > best_sim and sim >= self.coherence_threshold:
                    best_sim = sim
                    best_idx = i

            if best_idx < 0:
                break

            sequence.append(episodes[best_idx])
            used.add(best_idx)

        if len(sequence) >= self.min_sequence_length:
            self._ripple_count += 1
            self._total_memories_replayed += len(sequence)

            # Apply replay direction
            if direction == ReplayDirection.REVERSE:
                sequence = list(reversed(sequence))
                self._reverse_replays += 1
            elif direction == ReplayDirection.FORWARD:
                self._forward_replays += 1
            elif direction == ReplayDirection.BIDIRECTIONAL:
                # For bidirectional, we return forward but caller can process both
                self._forward_replays += 1
                self._reverse_replays += 1

            return sequence, direction

        return [], direction

    def _get_embedding(self, episode: Any) -> np.ndarray | None:
        """Extract embedding from episode object."""
        emb = getattr(episode, "embedding", None)
        if emb is None:
            emb = getattr(episode, "vector", None)
        if emb is not None:
            return np.asarray(emb)
        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> dict:
        """Get ripple statistics including replay direction."""
        return {
            "ripple_count": self._ripple_count,
            "total_memories_replayed": self._total_memories_replayed,
            "compression_factor": self.compression_factor,
            "forward_replays": self._forward_replays,
            "reverse_replays": self._reverse_replays,
            "reverse_ratio": (
                self._reverse_replays / max(1, self._forward_replays + self._reverse_replays)
            ),
            "default_direction": self.default_direction.value,
        }


class EpisodicMemory(Protocol):
    """Protocol for episodic memory."""

    async def get_recent(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> list[Any]:
        """Get recent episodes."""
        ...

    async def get_by_id(self, episode_id: UUID) -> Any:
        """Get episode by ID."""
        ...

    async def sample_random(
        self,
        limit: int = 50,
        session_filter: str | None = None,
        exclude_hours: int = 24,
    ) -> list[Any]:
        """Sample random episodes for interleaved replay (P3.4)."""
        ...


class SemanticMemory(Protocol):
    """Protocol for semantic memory."""

    async def create_or_strengthen(
        self,
        name: str,
        description: str,
        source_episode_id: UUID | None = None
    ) -> Any:
        """Create or strengthen entity."""
        ...

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        summary: str,
        details: str,
        embedding: list[float],
        source: str,
    ) -> Any:
        """Create new semantic entity (for REM abstraction)."""
        ...

    async def create_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: str,
        initial_weight: float,
    ) -> Any:
        """Create relationship between entities."""
        ...

    async def cluster_embeddings(
        self,
        embeddings: list[np.ndarray],
        min_cluster_size: int = 3
    ) -> list[list[int]]:
        """Cluster embeddings."""
        ...


class GraphStore(Protocol):
    """Protocol for graph store."""

    async def get_node(
        self,
        node_id: str
    ) -> dict | None:
        """Get single node by ID."""
        ...

    async def get_relationships(
        self,
        node_id: str,
        direction: str = "out"
    ) -> list[dict]:
        """Get relationships."""
        ...

    async def update_relationship_weight(
        self,
        source_id: str,
        target_id: str,
        new_weight: float
    ) -> None:
        """Update relationship weight."""
        ...

    async def delete_relationship(
        self,
        source_id: str,
        target_id: str
    ) -> None:
        """Delete relationship."""
        ...

    async def get_all_nodes(
        self,
        label: str | None = None
    ) -> list[dict]:
        """Get all nodes."""
        ...


class SleepConsolidation:
    """
    Sleep-based memory consolidation.

    Simulates the memory consolidation that occurs during sleep:
    - NREM: Replay and strengthen high-value experiences
    - REM: Create abstract concepts from clusters
    - Prune: Remove weak connections

    The consolidation follows biological timescales:
    - NREM ~75% of cycle, REM ~25%
    - Multiple NREM-REM cycles per sleep session
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        graph_store: GraphStore,
        # NREM parameters
        replay_hours: int = 24,
        max_replays: int = 100,
        outcome_weight: float = 0.3,
        importance_weight: float = 0.2,
        recency_weight: float = 0.2,
        prediction_error_weight: float = 0.3,  # P1-1: Prediction error priority
        # REM parameters
        min_cluster_size: int = 3,
        abstraction_threshold: float = 0.7,
        # Prune parameters
        prune_threshold: float = 0.05,
        homeostatic_target: float = 10.0,
        # Cycle parameters
        nrem_cycles: int = 4,
        # P2.5: Biologically accurate replay timing
        # Hippocampal replay occurs ~1-2 Hz during SWRs (500-1000ms intervals)
        # Previous 10ms was too fast and not biologically plausible
        replay_delay_ms: int = 500,
        # P3.4: Interleaved Replay (CLS Theory)
        interleave_enabled: bool = True,
        recent_ratio: float = 0.6,
        replay_batch_size: int = 100,
        # P7: VAE-based generative replay
        vae_enabled: bool = True,
        vae_latent_dim: int = 128,
        embedding_dim: int = 1024,
        # ATOM-P2-19: Seedable consolidation for reproducibility
        seed: int | None = None,
    ):
        """
        Initialize sleep consolidation.

        Args:
            episodic_memory: Episodic memory service
            semantic_memory: Semantic memory service
            graph_store: Graph store for relationships
            replay_hours: Hours of episodes to consider for replay
            max_replays: Maximum episodes to replay per cycle
            outcome_weight: Weight for outcome in priority scoring
            importance_weight: Weight for importance in priority scoring
            recency_weight: Weight for recency in priority scoring
            prediction_error_weight: Weight for prediction error magnitude (P1-1)
            min_cluster_size: Minimum cluster size for abstraction
            abstraction_threshold: Confidence threshold for abstraction
            prune_threshold: Weight threshold for pruning
            homeostatic_target: Target total weight per node
            nrem_cycles: Number of NREM cycles per sleep session
            replay_delay_ms: Delay between replays in milliseconds (default 500ms).
                P2.5: Changed from 10ms to 500ms for biological accuracy.
                Hippocampal sharp-wave ripples occur at ~1-2 Hz frequency.
            interleave_enabled: Enable CLS-style interleaved replay (P3.4)
            recent_ratio: Ratio of recent vs older memories (0.6 = 60% recent)
            replay_batch_size: Total batch size for interleaved replay
            vae_enabled: Enable VAE-based generative replay (P7)
            vae_latent_dim: Latent dimension for VAE generator
            embedding_dim: Embedding dimension (should match memory system)
            seed: Random seed for reproducibility (ATOM-P2-19)
        """
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.graph_store = graph_store

        # NREM parameters
        self.replay_hours = replay_hours
        self.max_replays = max_replays
        self.outcome_weight = outcome_weight
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.prediction_error_weight = prediction_error_weight  # P1-1

        # REM parameters
        self.min_cluster_size = min_cluster_size
        self.abstraction_threshold = abstraction_threshold

        # Prune parameters
        self.prune_threshold = prune_threshold
        self.homeostatic_target = homeostatic_target

        # Cycle parameters
        self.nrem_cycles = nrem_cycles
        self.replay_delay_ms = replay_delay_ms

        # P3.4: Interleaved replay parameters (CLS theory)
        self.interleave_enabled = interleave_enabled
        self.recent_ratio = recent_ratio
        self.replay_batch_size = replay_batch_size

        # ATOM-P2-19: Seedable RNG for reproducibility
        self._rng = np.random.default_rng(seed)

        # Sharp-wave ripple generator for compressed replay
        self.swr = SharpWaveRipple(
            compression_factor=10.0,
            min_sequence_length=3,
            max_sequence_length=8
        )

        # Optional homeostatic plasticity integration
        self._homeostatic: Any | None = None

        # P5.3: PlasticityManager for synaptic tagging and capture
        self._plasticity_manager: Any | None = None

        # P5.1/P7: GenerativeReplaySystem for wake-sleep consolidation (Hinton)
        # P7: Automatically initialize with VAE if enabled
        self._generative_replay: GenerativeReplaySystem | None = None
        self._vae_generator: VAEGenerator | None = None
        self.vae_enabled = vae_enabled

        if vae_enabled:
            # Create VAE generator with matching embedding dimension
            vae_config = VAEConfig(
                embedding_dim=embedding_dim,
                latent_dim=vae_latent_dim,
                hidden_dims=(512, 256),
                learning_rate=0.001,
                kl_weight=0.1,
            )
            self._vae_generator = VAEGenerator(vae_config)

            # Create GenerativeReplaySystem with VAE as generator
            replay_config = GenerativeReplayConfig(
                wake_learning_rate=0.01,
                sleep_learning_rate=0.001,
                n_sleep_samples=100,
                interleave_ratio=0.5,
            )
            self._generative_replay = GenerativeReplaySystem(
                config=replay_config,
                generator=self._vae_generator,
            )

            logger.info(
                f"P7: VAE-based generative replay initialized "
                f"(latent_dim={vae_latent_dim}, embedding_dim={embedding_dim})"
            )

        # P5: VAEReplayTrainer for training VAE from wake experiences
        self._vae_trainer: VAEReplayTrainer | None = None
        if vae_enabled and self._vae_generator is not None:
            self._vae_trainer = VAEReplayTrainer(
                vae=self._vae_generator,
                memory=self.episodic,
                config=VAETrainingConfig(
                    buffer_size=5000,
                    min_samples_for_training=32,
                    epochs_per_training=5,
                    batch_size=32,
                ),
            )
            logger.info("P5: VAEReplayTrainer initialized for wake-sleep training")

        # P5.4: GlymphaticConsolidationBridge for waste clearance during NREM
        self._glymphatic_bridge: Any | None = None

        # P5.5: SWRNeuralFieldCoupling for replay timing
        self._swr_coupling: Any | None = None

        # P7.3: DreamConsolidation for REM dream-based consolidation
        self._dream_consolidation: Any | None = None
        # Phase 1A: ReconsolidationEngine for embedding updates during replay
        self._reconsolidation_engine: Any | None = None


        # P1D: VTACircuit for RPE generation during replay
        self._vta_circuit: Any | None = None

        # State tracking
        self.current_phase = SleepPhase.WAKE
        self._replay_history: list[ReplayEvent] = []
        self._abstraction_history: list[AbstractionEvent] = []
        self._last_cycle_result: SleepCycleResult | None = None

    def set_homeostatic(self, homeostatic: Any) -> None:
        """
        Set homeostatic plasticity for synaptic scaling during sleep.

        Args:
            homeostatic: HomeostaticPlasticity instance
        """
        self._homeostatic = homeostatic

    def set_plasticity_manager(self, plasticity_manager: Any) -> None:
        """
        P5.3: Set PlasticityManager for synaptic tagging integration.

        The PlasticityManager handles:
        - Synaptic tagging during retrieval (tags created)
        - Tag capture during consolidation (tags become permanent)
        - Homeostatic scaling for network stability
        - LTD for competitive weakening

        Args:
            plasticity_manager: PlasticityManager instance from ww.learning.plasticity
        """
        self._plasticity_manager = plasticity_manager

    def set_generative_replay(self, replay: GenerativeReplaySystem | Any) -> None:
        """
        P5.1: Set GenerativeReplaySystem for wake-sleep consolidation.

        Implements Hinton's wake-sleep algorithm:
        - Wake phase: Learn from real experiences
        - Sleep phase: Generate synthetic samples for interleaved training
        - Prevents catastrophic forgetting via CLS interleaving

        Note: If vae_enabled=True (default), GenerativeReplaySystem with VAE
        is auto-initialized. Call this only to override with custom system.

        Args:
            replay: GenerativeReplaySystem instance from ww.learning.generative_replay
        """
        if self.vae_enabled and self._generative_replay is not None:
            logger.info(
                "P7: Overriding auto-initialized VAE replay with custom system"
            )
        self._generative_replay = replay

    def get_vae_statistics(self) -> dict | None:
        """
        P7: Get VAE generator statistics.

        Returns:
            VAE statistics dict or None if VAE not enabled
        """
        if self._vae_generator is not None:
            return self._vae_generator.get_statistics()
        return None

    async def train_vae_from_wake(
        self,
        n_samples: int = 100,
        epochs: int = 5,
    ) -> dict | None:
        """
        P5: Train VAE from recent wake experiences.

        Collects recent episodes and trains the VAE to generate
        realistic synthetic memories for interleaved replay.

        Args:
            n_samples: Number of recent episodes to collect
            epochs: Training epochs

        Returns:
            Training statistics or None if VAE not enabled
        """
        if self._vae_trainer is None:
            logger.warning("VAE trainer not initialized")
            return None

        # Collect wake samples
        collected = await self._vae_trainer.collect_wake_samples(n_samples=n_samples)
        if collected == 0:
            logger.debug("No samples collected for VAE training")
            return None

        # Train VAE
        stats = self._vae_trainer.train_vae(epochs=epochs)

        logger.info(
            f"P5: VAE trained from {collected} wake samples, "
            f"loss={stats.final_loss:.4f}"
        )

        return stats.to_dict()

    def generate_synthetic_memories(
        self,
        n_samples: int = 50,
        temperature: float = 0.8,
    ) -> list[np.ndarray]:
        """
        P5: Generate synthetic memories using trained VAE.

        Args:
            n_samples: Number of synthetic memories to generate
            temperature: Sampling temperature (lower = more realistic)

        Returns:
            List of synthetic embedding vectors
        """
        if self._vae_trainer is None:
            logger.warning("VAE trainer not initialized")
            return []

        return self._vae_trainer.generate_for_replay(
            n_samples=n_samples,
            temperature=temperature,
        )

    def get_vae_trainer_statistics(self) -> dict | None:
        """
        P5: Get VAE trainer statistics.

        Returns:
            Trainer statistics or None if not enabled
        """
        if self._vae_trainer is not None:
            return self._vae_trainer.get_statistics()
        return None

    def set_glymphatic_bridge(self, bridge: Any) -> None:
        """
        P5.4: Set GlymphaticConsolidationBridge for waste clearance protection.

        During NREM, protects currently-replaying memories from clearance
        while allowing clearance of inactive/weak memories.

        Args:
            bridge: GlymphaticConsolidationBridge from ww.nca.glymphatic_consolidation_bridge
        """
        self._glymphatic_bridge = bridge

    def set_swr_coupling(self, coupling: Any) -> None:
        """
        P5.5: Set SWRNeuralFieldCoupling for replay timing gating.

        Uses sharp-wave ripple timing to gate memory replay:
        - Replay only occurs during SWR windows
        - Temporal compression applied during ripples

        Args:
            coupling: SWRNeuralFieldCoupling from ww.nca.swr_coupling
        """
        self._swr_coupling = coupling

    def set_dream_consolidation(self, dream_consolidation: Any) -> None:
        """
        P7.3: Set DreamConsolidation for REM dream-based consolidation.

        During REM phase, uses DreamConsolidation to:
        - Generate dream trajectories from high-error episodes
        - Train predictor on dream sequences
        - Update replay priorities based on dream quality

        Args:
            dream_consolidation: DreamConsolidation from ww.dreaming.consolidation
        """
        self._dream_consolidation = dream_consolidation
        logger.info("P7.3: DreamConsolidation connected to SleepConsolidation")

    def set_vta_circuit(self, vta_circuit: Any) -> None:
        """
        P1D: Set VTACircuit for RPE generation during replay.

        During NREM replay, uses VTA to:
        - Generate RPE from replayed sequences
        - Provide credit assignment signals
        - Prioritize high-RPE sequences for replay

        Args:
            vta_circuit: VTACircuit from ww.nca.vta
        """
        self._vta_circuit = vta_circuit
        logger.info("P1D: VTACircuit connected to SleepConsolidation")

    def set_reconsolidation_engine(self, engine: Any) -> None:
        """
        Phase 1A: Set ReconsolidationEngine for embedding updates during replay.

        When set, sleep replay will update episode embeddings based on replay context.
        This implements biological reconsolidation where retrieved memories become
        labile and can be modified during the lability window.

        Args:
            engine: ReconsolidationEngine instance from ww.learning.reconsolidation
        """
        self._reconsolidation_engine = engine
        logger.info("Phase 1A: ReconsolidationEngine connected to SleepConsolidation")

    async def get_replay_batch(
        self,
        recent_ratio: float | None = None,
        batch_size: int | None = None,
    ) -> list[Any]:
        """
        Get interleaved batch of recent and older memories for replay (P3.4).

        Implements CLS (Complementary Learning Systems) theory principle:
        mixing recent experiences with older consolidated memories prevents
        catastrophic forgetting while still prioritizing new learning.

        Biological basis: During sleep, the hippocampus replays both
        recent experiences AND older memories in an interleaved fashion,
        which helps maintain previously learned knowledge while
        integrating new experiences.

        Args:
            recent_ratio: Override for ratio of recent memories (default: self.recent_ratio)
            batch_size: Override for batch size (default: self.replay_batch_size)

        Returns:
            Shuffled list combining recent and older episodes
        """
        import random

        ratio = recent_ratio if recent_ratio is not None else self.recent_ratio
        size = batch_size if batch_size is not None else self.replay_batch_size

        recent_count = int(size * ratio)
        old_count = size - recent_count

        logger.debug(
            f"get_replay_batch: batch_size={size}, recent_ratio={ratio}, "
            f"recent_count={recent_count}, old_count={old_count}"
        )

        # Get recent episodes
        try:
            recent = await self.episodic.get_recent(
                hours=self.replay_hours,
                limit=recent_count,
            )
        except Exception as e:
            logger.warning(f"Failed to get recent episodes: {e}")
            recent = []

        # Get older episodes (random sample)
        old = []
        if old_count > 0:
            try:
                old = await self.episodic.sample_random(
                    limit=old_count,
                    exclude_hours=self.replay_hours,  # Don't overlap with recent
                )
            except Exception as e:
                logger.warning(f"Failed to sample old episodes: {e}")
                old = []

        # Combine and shuffle
        combined = recent + old
        random.shuffle(combined)

        # Filter out procedural memories - these route through striatal MSN pathway, not hippocampal SWR
        # Biological basis: Motor/skill memories (procedural) consolidate via basal ganglia,
        # while episodic/semantic memories consolidate via hippocampal sharp-wave ripples
        combined = [m for m in combined if getattr(m, 'memory_type', 'episodic') != 'procedural']

        logger.info(
            f"Interleaved replay batch: {len(recent)} recent + {len(old)} old = "
            f"{len(combined)} total (procedural memories excluded)"
        )

        return combined

    async def get_procedural_replay_batch(
        self,
        batch_size: int | None = None,
    ) -> list[Any]:
        """
        Get batch of procedural memories for striatal MSN pathway replay.

        Biological basis: Procedural memories (motor skills, habits) consolidate
        via basal ganglia medium spiny neuron (MSN) pathways during sleep, not
        through hippocampal sharp-wave ripples. The striatal consolidation pathway
        uses distinct replay mechanisms optimized for motor sequence learning.

        This is a placeholder for future implementation. When implemented, this will:
        - Route procedural memories through basal ganglia MSN pathways
        - Use striatal replay patterns (different from hippocampal SWR)
        - Apply distinct consolidation rules for motor/skill memories

        Args:
            batch_size: Override for batch size (default: self.replay_batch_size)

        Returns:
            Empty list (placeholder - striatal pathway not yet implemented)
        """
        logger.debug(
            "Procedural memory replay not yet implemented - "
            "future: route through striatal MSN pathway"
        )
        return []

    async def _generate_replay_rpe(self, replay_sequence: list[Any]) -> list[float]:
        """
        P1D: Generate RPE from replayed sequence.

        Uses VTA circuit to compute temporal difference errors across
        the replay sequence. This provides credit assignment signals
        for learning during consolidation.

        Biological basis: During sleep replay, dopamine neurons show
        activity patterns related to reward prediction errors, even
        though no actual reward is present. This "offline" RPE helps
        strengthen appropriate associations.

        Args:
            replay_sequence: List of episodes in replay order

        Returns:
            List of RPE values for each transition
        """
        if not self._vta_circuit:
            return []

        rpes = []
        for i, episode in enumerate(replay_sequence):
            if i == 0:
                # First episode: no previous transition
                continue

            # Estimate value from episode importance/relevance
            prev_value = self._estimate_value(replay_sequence[i - 1])
            curr_value = self._estimate_value(episode)

            # Get reward signal (importance or outcome score)
            reward = getattr(episode, "importance", 0.5)
            outcome_score = getattr(episode, "outcome_score", None)
            if outcome_score is not None:
                reward = outcome_score

            # Compute RPE using VTA's outcome-based method
            # RPE = actual - expected
            expected = prev_value  # Simple expectation from previous value
            rpe = self._vta_circuit.compute_rpe_from_outcome(
                actual_outcome=reward,
                expected_outcome=expected
            )

            rpes.append(rpe)

            # Feed RPE to VTA for processing (updates DA, eligibility)
            self._vta_circuit.process_rpe(rpe, dt=0.1)

        return rpes

    def _estimate_value(self, episode: Any) -> float:
        """
        P1D: Estimate value of an episode for RPE computation.

        Combines multiple factors to estimate expected future value:
        - Importance (emotional valence)
        - Outcome score
        - Relevance/retrieval count

        Args:
            episode: Episode to estimate value for

        Returns:
            Estimated value [0, 1]
        """
        # Get importance
        importance = getattr(episode, "importance", 0.5)
        if importance is None:
            importance = 0.5

        # Get outcome score
        outcome = getattr(episode, "outcome_score", 0.5)
        if outcome is None:
            outcome = 0.5

        # Get relevance (retrieval count as proxy)
        relevance = getattr(episode, "retrieval_count", 0)
        relevance_score = min(relevance / 10.0, 1.0)  # Normalize to [0, 1]

        # Weighted combination
        value = (
            0.4 * importance +
            0.4 * outcome +
            0.2 * relevance_score
        )

        return float(np.clip(value, 0.0, 1.0))

    def _prioritize_by_rpe(
        self,
        sequences: list[list[Any]],
        rpes: list[list[float]]
    ) -> list[list[Any]]:
        """
        P1D: Weight replay probability by cumulative RPE.

        High RPE sequences are surprising and should be replayed
        more frequently for better learning.

        Args:
            sequences: List of replay sequences
            rpes: Corresponding RPE lists for each sequence

        Returns:
            Sequences sorted by RPE priority (highest first)
        """
        if not rpes or not sequences:
            return sequences

        # Compute priority scores (use absolute cumulative RPE)
        priorities = []
        for rpe_seq in rpes:
            if rpe_seq:
                # Use mean absolute RPE as priority
                priority = float(np.mean([abs(r) for r in rpe_seq]))
            else:
                priority = 0.0
            priorities.append(priority)

        # Sort sequences by priority (descending)
        sorted_pairs = sorted(
            zip(sequences, priorities),
            key=lambda x: x[1],
            reverse=True
        )

        return [seq for seq, _ in sorted_pairs]

    async def nrem_phase(
        self,
        session_id: str,
        replay_count: int | None = None
    ) -> list[ReplayEvent]:
        """
        Execute NREM (slow-wave sleep) phase.

        Replays high-value experiences to strengthen hippocampal → neocortical
        transfer. When interleave_enabled=True (P3.4), mixes recent and older
        memories per CLS theory to prevent catastrophic forgetting.

        P1D: Generates RPE from replay sequences via VTA for credit assignment.

        Args:
            session_id: Session identifier
            replay_count: Override max replays

        Returns:
            List of replay events
        """
        self.current_phase = SleepPhase.NREM
        max_replays = replay_count or self.max_replays

        # P3.4: Use interleaved replay if enabled (CLS theory)
        if self.interleave_enabled:
            try:
                to_replay = await self.get_replay_batch(batch_size=max_replays)
                if to_replay:
                    # Prioritize the interleaved batch
                    to_replay = self._prioritize_for_replay(to_replay)[:max_replays]
                    logger.info(
                        f"NREM using interleaved replay: {len(to_replay)} episodes "
                        f"(ratio={self.recent_ratio})"
                    )
            except Exception as e:
                logger.warning(f"Interleaved replay failed, falling back: {e}")
                to_replay = None
        else:
            to_replay = None

        # Fallback: original behavior (recent-only)
        if not to_replay:
            try:
                recent = await self.episodic.get_recent(
                    hours=self.replay_hours,
                    limit=max_replays * 2  # Get extra for filtering
                )
            except Exception as e:
                logger.error(f"Failed to get recent episodes: {e}")
                return []

            if not recent:
                logger.debug("No recent episodes for NREM replay")
                return []

            # Compute priority scores
            prioritized = self._prioritize_for_replay(recent)

            # Take top episodes
            to_replay = prioritized[:max_replays]

        # Generate SWR sequences for compressed replay
        events = []
        processed_ids = set()

        # P5.5: SWR timing gates memory replay (Buzsaki 2015, Girardeau 2009)
        # Replay should only occur during sharp-wave ripple events
        swr_gated_replays = 0
        swr_blocked_replays = 0

        # P1D: Track sequences and RPEs for priority-based replay
        replay_sequences = []
        sequence_rpes = []

        # Try SWR-based compressed replay first
        swr_worked = False
        for ripple_num in range(min(5, len(to_replay) // 3 + 1)):
            # generate_ripple_sequence returns (sequence, direction) tuple
            ripple_seq, direction = self.swr.generate_ripple_sequence(to_replay)

            if ripple_seq:
                swr_worked = True

                # P1D: Generate RPE from this sequence
                if self._vta_circuit is not None:
                    try:
                        rpes = await self._generate_replay_rpe(ripple_seq)
                        replay_sequences.append(ripple_seq)
                        sequence_rpes.append(rpes)

                        if rpes:
                            logger.debug(
                                f"P1D: Replay RPE for sequence {ripple_num}: "
                                f"mean={np.mean(rpes):.3f}, std={np.std(rpes):.3f}"
                            )
                    except Exception as e:
                        logger.warning(f"P1D: RPE generation failed: {e}")
                        sequence_rpes.append([])

                for pos, episode in enumerate(ripple_seq):
                    ep_id = getattr(episode, "id", None)
                    if ep_id and ep_id in processed_ids:
                        continue

                    # P5.5: Check SWR coupling for timing gate
                    swr_allowed = True
                    if self._swr_coupling is not None:
                        try:
                            # Step the SWR coupling to advance dynamics
                            swr_occurred = self._swr_coupling.step(dt=0.01)

                            # Only replay during RIPPLING phase
                            from ww.nca.swr_coupling import SWRPhase
                            current_phase = getattr(
                                getattr(self._swr_coupling, 'state', None),
                                'phase', None
                            )

                            if current_phase not in [SWRPhase.INITIATING, SWRPhase.RIPPLING]:
                                swr_allowed = False
                                swr_blocked_replays += 1
                            else:
                                swr_gated_replays += 1
                        except Exception as e:
                            logger.debug(f"SWR coupling check failed: {e}")
                            # Continue with replay if check fails

                    if not swr_allowed:
                        continue  # Skip this replay - not in SWR window

                    try:
                        # Get RPE for this episode if available
                        episode_rpe = 0.0
                        if sequence_rpes and pos > 0 and len(sequence_rpes[-1]) >= pos:
                            # RPE at index pos-1 represents transition TO this episode
                            episode_rpe = sequence_rpes[-1][pos - 1]

                        event = await self._replay_episode(episode)
                        if event:
                            event.direction = direction
                            event.sequence_position = pos
                            event.rpe = episode_rpe  # P1D: Store RPE
                            events.append(event)
                            self._replay_history.append(event)
                            if ep_id:
                                processed_ids.add(ep_id)

                        # Simulate biological timing (compressed by SWR)
                        if self.replay_delay_ms > 0:
                            await asyncio.sleep(
                                self.replay_delay_ms / 1000 / self.swr.compression_factor
                            )

                    except Exception as e:
                        logger.warning(f"Failed to replay episode: {e}")

        # Fallback: if SWR couldn't generate sequences (e.g., no embeddings),
        # replay episodes directly in priority order
        if not swr_worked and to_replay:
            logger.debug("SWR unavailable, using direct replay fallback")
            for episode in to_replay:
                ep_id = getattr(episode, "id", None)
                if ep_id and ep_id in processed_ids:
                    continue

                try:
                    event = await self._replay_episode(episode)
                    if event:
                        events.append(event)
                        self._replay_history.append(event)
                        if ep_id:
                            processed_ids.add(ep_id)

                    if self.replay_delay_ms > 0:
                        await asyncio.sleep(self.replay_delay_ms / 1000)

                except Exception as e:
                    logger.warning(f"Failed to replay episode: {e}")

        # P1D: Log RPE statistics
        if sequence_rpes and self._vta_circuit is not None:
            all_rpes = [rpe for rpe_list in sequence_rpes for rpe in rpe_list]
            if all_rpes:
                logger.info(
                    f"P1D: VTA active during consolidation - "
                    f"sequences={len(sequence_rpes)}, "
                    f"mean_rpe={np.mean(all_rpes):.3f}, "
                    f"std_rpe={np.std(all_rpes):.3f}"
                )

        # Collect embeddings from replayed episodes for homeostatic and generative replay
        embeddings: list[np.ndarray] = []
        if events:
            for event in events:
                try:
                    ep = await self.episodic.get_by_id(event.episode_id)
                    if ep:
                        emb = getattr(ep, "embedding", None)
                        if emb is not None:
                            embeddings.append(np.asarray(emb))
                except Exception:
                    pass

        # Apply homeostatic scaling if available
        if self._homeostatic and embeddings:
            try:
                emb_array = np.array(embeddings)
                self._homeostatic.update_statistics(emb_array)
                if self._homeostatic.needs_scaling():
                    # Log scaling event
                    logger.debug(
                        f"Homeostatic scaling applied during NREM: "
                        f"mean_norm={self._homeostatic.get_state().mean_norm:.3f}"
                    )
            except Exception as e:
                logger.debug(f"Homeostatic integration skipped: {e}")

        # P5.1/P7: Generative replay for wake-sleep consolidation (Hinton)
        # Feed real embeddings to wake phase, generate synthetic for interleaving
        if self._generative_replay and embeddings:
            try:
                # Wake phase: process real episode embeddings
                await self._generative_replay.process_wake(embeddings)

                # Sleep phase: generate synthetic samples
                n_synthetic = max(1, len(embeddings) // 2)
                sleep_stats = await self._generative_replay.run_sleep_phase(
                    n_samples=n_synthetic
                )

                # Get interleaved batch for future training
                # This prevents catastrophic forgetting by mixing old and new
                interleaved_batch = self._generative_replay.get_interleaved_batch(
                    batch_size=len(embeddings),
                    new_embeddings=embeddings
                )

                logger.debug(
                    f"P5.1: Generative replay - wake samples: {len(embeddings)}, "
                    f"synthetic generated: {sleep_stats.n_samples_processed}, "
                    f"interleaved batch: {len(interleaved_batch)}"
                )
            except Exception as e:
                logger.warning(f"Generative replay integration failed: {e}")

        # P5.3: Capture synaptic tags during consolidation
        # Tags created during retrieval are "captured" by protein synthesis
        # during sleep, making them permanent (late-phase LTP)
        tags_captured = 0
        if self._plasticity_manager and events:
            try:
                # Get all entity IDs that were involved in replay
                replayed_entity_ids = []
                for event in events:
                    try:
                        ep = await self.episodic.get_by_id(event.episode_id)
                        if ep:
                            # Could extract entities from episode content
                            replayed_entity_ids.append(str(event.episode_id))
                    except Exception:
                        pass

                # Capture tags (simulates protein synthesis during sleep)
                result = await self._plasticity_manager.on_consolidation(
                    all_entity_ids=replayed_entity_ids,
                    store=self.graph_store,
                )
                tags_captured = result.get("tags_captured", 0)
                logger.debug(
                    f"P5.3: Captured {tags_captured} synaptic tags during NREM consolidation"
                )
            except Exception as e:
                logger.warning(f"Synaptic tag capture failed: {e}")

        # P5.5: Log SWR gating statistics
        if self._swr_coupling is not None:
            logger.debug(
                f"P5.5: SWR timing gating - allowed={swr_gated_replays}, "
                f"blocked={swr_blocked_replays}"
            )

        logger.info(
            f"NREM phase complete: {len(events)} episodes replayed, "
            f"{tags_captured} tags captured for session {session_id}"
        )

        return events

    async def rem_phase(
        self,
        session_id: str
    ) -> list[AbstractionEvent]:
        """
        Execute REM phase.

        Creates abstract concepts by clustering recent semantic
        entities and finding patterns.

        Args:
            session_id: Session identifier

        Returns:
            List of abstraction events
        """
        self.current_phase = SleepPhase.REM

        # Get recent semantic entities
        try:
            nodes = await self.graph_store.get_all_nodes(label="Entity")
        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            return []

        if len(nodes) < self.min_cluster_size:
            logger.debug("Not enough entities for REM abstraction")
            return []

        # Extract embeddings
        embeddings = []
        node_ids = []
        for node in nodes:
            emb = node.get("embedding")
            if emb is None:
                emb = node.get("properties", {}).get("embedding")
            if emb is not None:
                embeddings.append(np.asarray(emb))
                node_ids.append(node.get("id"))

        if len(embeddings) < self.min_cluster_size:
            return []

        # Convert to numpy array for efficient operations
        embeddings_array = np.array(embeddings)

        # P7.3: Run dream consolidation if available
        # Dreams use high-error episodes to generate training trajectories
        dream_results = None
        if self._dream_consolidation is not None:
            try:
                # Prepare episode data for dreaming
                recent_episodes = [
                    (UUID(str(nid)), embeddings_array[i])
                    for i, nid in enumerate(node_ids) if nid
                ]

                # Get high-error episodes from replay history
                high_error = [
                    (e.episode_id, embeddings_array[node_ids.index(str(e.episode_id))])
                    for e in self._replay_history[-50:]
                    if hasattr(e, 'episode_id') and str(e.episode_id) in node_ids
                ][:10]

                # Run dream cycle
                dream_results = self._dream_consolidation.run_dream_cycle(
                    recent_episodes=recent_episodes[:20],
                    reference_embeddings=list(embeddings_array),
                    high_error_episodes=high_error if high_error else None,
                )

                logger.debug(
                    f"P7.3: Dream cycle complete - "
                    f"dreams={dream_results.n_dreams}, "
                    f"quality={dream_results.mean_quality:.3f}"
                )

            except Exception as e:
                logger.warning(f"P7.3: Dream consolidation failed: {e}")

        # Cluster embeddings
        try:
            clusters = await self._cluster_embeddings(embeddings_array)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return []

        # Create abstractions from clusters
        events = []
        for cluster_indices in clusters:
            if len(cluster_indices) >= self.min_cluster_size:
                cluster_ids = [node_ids[i] for i in cluster_indices if i < len(node_ids)]

                try:
                    event = await self._create_abstraction(cluster_ids, embeddings_array, cluster_indices)
                    if event:
                        events.append(event)
                        self._abstraction_history.append(event)
                except Exception as e:
                    logger.warning(f"Failed to create abstraction: {e}")

        logger.info(
            f"REM phase complete: {len(events)} abstractions created "
            f"for session {session_id}"
        )

        return events

    async def prune_phase(self) -> tuple[int, int]:
        """
        Execute pruning phase (synaptic downscaling).

        Removes weak connections and applies homeostatic scaling.
        P5.4: Also runs glymphatic waste clearance during NREM-like prune phase.

        Returns:
            Tuple of (pruned_count, strengthened_count)
        """
        self.current_phase = SleepPhase.PRUNE

        pruned = 0
        strengthened = 0

        # P5.4: Run glymphatic clearance during prune phase
        # Biological basis: Xie et al. 2013 - waste clearance peaks during NREM
        glymphatic_cleared = 0
        if self._glymphatic_bridge is not None:
            try:
                # Simulate NREM-deep conditions for maximal clearance
                waste_state = self._glymphatic_bridge.step(
                    wake_sleep_mode="nrem_deep",
                    delta_up_state=True,
                    ne_level=0.1,  # Low NE during deep NREM
                    ach_level=0.05,  # Low ACh (not REM)
                    dt=1.0
                )
                glymphatic_cleared = getattr(waste_state, 'total_cleared', 0)
                logger.info(
                    f"P5.4: Glymphatic clearance during prune: "
                    f"cleared={glymphatic_cleared} items, "
                    f"rate={getattr(waste_state, 'clearance_rate', 0):.2f}"
                )
            except Exception as e:
                logger.warning(f"Glymphatic clearance failed: {e}")

        try:
            nodes = await self.graph_store.get_all_nodes()
        except Exception as e:
            logger.error(f"Failed to get nodes for pruning: {e}")
            return 0, 0

        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue

            try:
                rels = await self.graph_store.get_relationships(node_id, "out")
            except Exception:
                continue

            if not rels:
                continue

            # Compute total weight
            total_weight = sum(
                r.get("properties", {}).get("weight", 0.5)
                for r in rels
            )

            for rel in rels:
                weight = rel.get("properties", {}).get("weight", 0.5)
                other_id = rel.get("other_id")

                if not other_id:
                    continue

                # Prune weak connections
                if weight < self.prune_threshold:
                    try:
                        await self.graph_store.delete_relationship(
                            source_id=node_id,
                            target_id=other_id
                        )
                        pruned += 1
                    except Exception:
                        pass

                # Apply homeostatic scaling if total too high
                elif total_weight > self.homeostatic_target * 1.2:
                    scale = self.homeostatic_target / total_weight
                    new_weight = weight * scale
                    try:
                        await self.graph_store.update_relationship_weight(
                            source_id=node_id,
                            target_id=other_id,
                            new_weight=new_weight
                        )
                        strengthened += 1  # Really "scaled" but track it
                    except Exception:
                        pass

        logger.info(
            f"Prune phase complete: {pruned} pruned, "
            f"{strengthened} scaled"
        )

        return pruned, strengthened

    async def full_sleep_cycle(
        self,
        session_id: str
    ) -> SleepCycleResult:
        """
        Execute complete sleep cycle with alternating NREM-REM phases.

        Biological sleep consists of ~4-5 cycles of NREM → REM,
        with a final pruning phase.

        Args:
            session_id: Session identifier

        Returns:
            Summary of sleep cycle results
        """
        start_time = datetime.now()
        total_replays = 0
        total_abstractions = 0

        logger.info(f"Starting sleep cycle for session {session_id}")

        # Execute NREM-REM cycles
        for cycle in range(self.nrem_cycles):
            logger.debug(f"Sleep cycle {cycle + 1}/{self.nrem_cycles}")

            # NREM phase (longer)
            replays = await self.nrem_phase(
                session_id,
                replay_count=self.max_replays // self.nrem_cycles
            )
            total_replays += len(replays)

            # REM phase (shorter, less frequent early)
            if cycle >= 1:  # REM gets longer in later cycles
                abstractions = await self.rem_phase(session_id)
                total_abstractions += len(abstractions)

        # Final pruning phase
        pruned, strengthened = await self.prune_phase()

        self.current_phase = SleepPhase.WAKE
        end_time = datetime.now()

        result = SleepCycleResult(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            nrem_replays=total_replays,
            rem_abstractions=total_abstractions,
            pruned_connections=pruned,
            strengthened_connections=strengthened,
            total_duration_seconds=(end_time - start_time).total_seconds()
        )

        self._last_cycle_result = result

        logger.info(
            f"Sleep cycle complete: {total_replays} replays, "
            f"{total_abstractions} abstractions, {pruned} pruned"
        )

        return result

    def _prioritize_for_replay(
        self,
        episodes: list[Any]
    ) -> list[Any]:
        """
        Prioritize episodes for replay based on value and prediction error.

        P1-1: Added prediction error to prioritization formula.

        Value = outcome_weight * outcome + importance_weight * importance
                + recency_weight * recency + prediction_error_weight * |PE|

        High prediction error memories are surprising and thus more valuable
        for replay (World Models insight: learn from what you couldn't predict).

        Args:
            episodes: List of episodes

        Returns:
            Sorted list (highest priority first)
        """
        now = datetime.now()

        def compute_priority(episode) -> float:
            # Get outcome score (0-1)
            outcome = getattr(episode, "outcome_score", 0.5)
            if hasattr(episode, "outcome"):
                outcome_val = episode.outcome
                if hasattr(outcome_val, "value"):
                    # Enum to score
                    outcome_map = {"success": 1.0, "partial": 0.7, "neutral": 0.5, "failure": 0.2}
                    outcome = outcome_map.get(outcome_val.value, 0.5)
                elif isinstance(outcome_val, (int, float)):
                    outcome = float(outcome_val)

            # Get importance (emotional valence)
            importance = getattr(episode, "emotional_valence", 0.5)
            if importance is None:
                importance = 0.5

            # Get recency (decay over 24 hours)
            created = getattr(episode, "created_at", None)
            if created is None:
                created = getattr(episode, "timestamp", now)
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created = now

            if hasattr(created, "tzinfo") and created.tzinfo is not None:
                created = created.replace(tzinfo=None)

            age_hours = (now - created).total_seconds() / 3600
            recency = max(0.0, 1.0 - age_hours / self.replay_hours)

            # P1-1: Get prediction error magnitude
            # High |PE| = surprising = more valuable for replay
            pe = getattr(episode, "prediction_error", None)
            if pe is None:
                pe = 0.0  # No PE data defaults to baseline priority
            else:
                pe = abs(float(pe))  # Use magnitude
                pe = min(pe, 1.0)  # Clamp to [0, 1]

            return (
                self.outcome_weight * outcome +
                self.importance_weight * importance +
                self.recency_weight * recency +
                self.prediction_error_weight * pe
            )

        # Sort by priority (descending)
        return sorted(episodes, key=compute_priority, reverse=True)

    async def _replay_episode(
        self,
        episode: Any
    ) -> ReplayEvent | None:
        """
        Replay a single episode to strengthen semantic connections.

        Args:
            episode: Episode to replay

        Returns:
            ReplayEvent if successful
        """
        episode_id = getattr(episode, "id", None)
        if episode_id is None:
            return None

        # Extract key entities from episode content
        getattr(episode, "content", "")
        context = getattr(episode, "context", None)

        # Simple entity extraction (could be enhanced with NER)
        entities = []
        if context:
            # Extract from context
            if hasattr(context, "project"):
                entities.append(f"project:{context.project}")
            if hasattr(context, "file"):
                entities.append(f"file:{context.file}")
            if hasattr(context, "tool"):
                entities.append(f"tool:{context.tool}")

        # Strengthen semantic connections
        strengthened = False
        for entity_ref in entities:
            try:
                await self.semantic.create_or_strengthen(
                    name=entity_ref,
                    description="Entity from episode replay",
                    source_episode_id=episode_id
                )
                strengthened = True
            except Exception as e:
                logger.debug(f"Failed to strengthen entity {entity_ref}: {e}")


        # Phase 1A: Update embeddings via reconsolidation if configured
        if self._reconsolidation_engine is not None:
            try:
                # Import lability module to check window
                from ww.consolidation.lability import is_reconsolidation_eligible

                # Check if episode has last_accessed (retrieval time) for lability window
                # ATOM-P1-8: Require retrieval before reconsolidation (no last_accessed = not retrieved)
                last_retrieval = getattr(episode, "last_accessed", None)
                if last_retrieval is None:
                    # No retrieval timestamp - memory hasn't been retrieved yet
                    # Cannot reconsolidate until retrieved (biological requirement)
                    can_reconsolidate = False
                else:
                    # Check if within lability window
                    can_reconsolidate = is_reconsolidation_eligible(
                        last_retrieval,
                        window_hours=6.0  # Standard lability window
                    )

                if can_reconsolidate:
                    # Get episode embedding
                    episode_emb = getattr(episode, "embedding", None)
                    if episode_emb is not None:
                        episode_emb = np.asarray(episode_emb)

                        # Create query embedding as average of recent episodes
                        # (simulates retrieval context)
                        recent_episodes = await self.episodic.get_recent(hours=24, limit=10)
                        query_embs = []
                        for ep in recent_episodes:
                            emb = getattr(ep, "embedding", None)
                            if emb is not None:
                                query_embs.append(np.asarray(emb))

                        if query_embs:
                            query_emb = np.mean(query_embs, axis=0)
                            # Normalize query embedding
                            query_norm = np.linalg.norm(query_emb)
                            if query_norm > 0:
                                query_emb = query_emb / query_norm

                            # Compute outcome score for reconsolidation
                            outcome_score = getattr(episode, "outcome_score", 0.5)

                            # Call reconsolidation engine
                            new_embedding = self._reconsolidation_engine.reconsolidate(
                                memory_id=UUID(str(episode_id)),
                                memory_embedding=episode_emb,
                                query_embedding=query_emb,
                                outcome_score=outcome_score
                            )

                            if new_embedding is not None:
                                # Phase 1.2 FIX: Persist reconsolidated embedding to vector store
                                try:
                                    updated_count = await self.episodic.vector_store.batch_update_vectors(
                                        collection=self.episodic.vector_store.episodes_collection,
                                        updates=[(str(episode_id), new_embedding.tolist())],
                                    )
                                    logger.info(
                                        f"Phase 1A: Reconsolidated and PERSISTED episode {episode_id} "
                                        f"during NREM replay (updated={updated_count})"
                                    )
                                except Exception as persist_err:
                                    logger.warning(
                                        f"Phase 1A: Reconsolidation computed but persistence failed for "
                                        f"{episode_id}: {persist_err}"
                                    )
                else:
                    logger.debug(
                        f"Episode {episode_id} outside lability window, "
                        "skipping reconsolidation"
                    )

            except Exception as e:
                logger.debug(f"Phase 1A: Reconsolidation failed for {episode_id}: {e}")

        return ReplayEvent(
            episode_id=episode_id,
            priority_score=0.0,  # Could compute
            strengthening_applied=strengthened,
            entities_extracted=entities
        )

    async def _cluster_embeddings(
        self,
        embeddings: np.ndarray
    ) -> list[list[int]]:
        """
        Cluster embeddings for REM abstraction.

        Uses simple k-means or DBSCAN-style clustering.

        Args:
            embeddings: Array of embeddings

        Returns:
            List of cluster index lists
        """
        if len(embeddings) < self.min_cluster_size:
            return []

        # Simple cosine similarity clustering
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Compute similarity matrix
        sim_matrix = normalized @ normalized.T

        # Greedy clustering based on similarity
        clusters = []
        used = set()

        for i in range(len(embeddings)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(embeddings)):
                if j in used:
                    continue
                if sim_matrix[i, j] > self.abstraction_threshold:
                    cluster.append(j)
                    used.add(j)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        return clusters

    async def _generate_concept_name(
        self,
        entity_names: list[str],
        entity_types: list[str],
    ) -> str:
        """
        Generate a descriptive name for an abstract concept.

        Uses entity types and names to create a meaningful concept name.
        Follows biological inspiration: REM sleep creates generalizations
        from specific instances.

        Args:
            entity_names: Names of entities in cluster
            entity_types: Types of entities in cluster

        Returns:
            Generated concept name
        """
        from collections import Counter

        # Count entity types
        type_counts = Counter(entity_types)
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else "CONCEPT"

        # Find common words in entity names
        if not entity_names:
            return f"Abstract {dominant_type.title()}"

        # Tokenize and find common terms
        all_words: list[str] = []
        for name in entity_names:
            words = name.lower().split()
            all_words.extend(words)

        word_counts = Counter(all_words)

        # Filter out common stopwords
        stopwords = {"the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are"}
        significant_words = [
            word for word, count in word_counts.most_common(10)
            if word not in stopwords and len(word) > 2 and count >= 2
        ]

        # Build concept name
        if significant_words:
            # Take top 2-3 significant words
            theme_words = significant_words[:3]
            theme = " ".join(w.title() for w in theme_words)
            return f"{theme} ({dominant_type.title()}s)"
        else:
            # Fallback: use dominant type and count
            return f"Related {dominant_type.title()}s ({len(entity_names)})"

    async def _create_abstraction(
        self,
        cluster_ids: list[str],
        embeddings: np.ndarray,
        cluster_indices: list[int]
    ) -> AbstractionEvent | None:
        """
        Create and persist an abstract concept from a cluster.

        P5.1: Stores abstract concepts from REM sleep.
        - Generates meaningful concept name from cluster members
        - Creates Entity with type=CONCEPT
        - Persists to semantic memory
        - Creates ABSTRACTS relationships to source entities

        Args:
            cluster_ids: Node IDs in cluster
            embeddings: All embeddings
            cluster_indices: Indices into embeddings

        Returns:
            AbstractionEvent with entity_id if successful, None otherwise
        """
        if len(cluster_indices) < self.min_cluster_size:
            return None

        # Compute cluster centroid
        cluster_embs = embeddings[cluster_indices]
        centroid = np.mean(cluster_embs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        else:
            return None

        # ATOM-P2-12: Semantic coherence validation for abstractions
        # Verify minimum coherence of ALL cluster members (not just average)
        if len(cluster_embs) > 1:
            for emb in cluster_embs:
                cos_sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb) + 1e-8)
                if cos_sim < 0.3:  # Minimum coherence threshold
                    logger.warning(f"Incoherent cluster member: cos_sim={cos_sim:.3f} < 0.3")
                    return None  # Reject this abstraction

        # ATOM-P0-6: Add provenance tracking to consolidation outputs
        from ww.core.provenance import sign_embedding

        provenance = sign_embedding(
            centroid, origin="rem_abstraction", creator_id="consolidation.sleep"
        )

        # Compute confidence (average similarity to centroid)
        sims = cluster_embs @ centroid
        confidence = float(np.mean(sims))

        if confidence < self.abstraction_threshold:
            return None

        # Get entity details for concept name generation
        entity_names: list[str] = []
        entity_types: list[str] = []
        source_entity_ids: list[UUID] = []

        for node_id in cluster_ids:
            try:
                node = await self.graph_store.get_node(node_id)
                if node:
                    props = node.get("properties", node)
                    name = props.get("name", "")
                    entity_type = props.get("entity_type", "CONCEPT")
                    if name:
                        entity_names.append(name)
                    entity_types.append(entity_type)
                    # Try to extract UUID from node_id
                    try:
                        source_entity_ids.append(UUID(node_id))
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                logger.debug(f"Failed to get node {node_id}: {e}")

        # Generate concept name
        concept_name = await self._generate_concept_name(entity_names, entity_types)

        # Create summary from member entities
        if entity_names:
            summary = f"Abstract concept linking: {', '.join(entity_names[:5])}"
            if len(entity_names) > 5:
                summary += f" and {len(entity_names) - 5} more"
        else:
            summary = f"Abstract concept from {len(cluster_ids)} related entities"

        # Persist to semantic memory
        entity_id: UUID | None = None
        try:
            # ATOM-P0-6: Include provenance in entity metadata
            entity = await self.semantic.create_entity(
                name=concept_name,
                entity_type="CONCEPT",
                summary=summary,
                details=f"REM abstraction with confidence {confidence:.3f}. "
                        f"Source entities: {len(cluster_ids)}. "
                        f"Provenance: hash={provenance.content_hash[:16]}..., "
                        f"record_id={provenance.record_id}",
                embedding=centroid.tolist(),
                source="rem_abstraction",
            )
            entity_id = entity.id
            logger.info(
                f"Created abstract concept: {concept_name} (id={entity_id}, "
                f"provenance={provenance.record_id})"
            )

            # Create ABSTRACTS relationships to source entities
            for source_id in source_entity_ids:
                try:
                    await self.semantic.create_relationship(
                        source_id=entity_id,
                        target_id=source_id,
                        relation_type="ABSTRACTS",
                        initial_weight=confidence,
                    )
                except Exception as e:
                    logger.debug(f"Failed to create relationship to {source_id}: {e}")

        except Exception as e:
            logger.warning(f"Failed to persist abstract concept: {e}")
            # Continue without persistence - still return event

        return AbstractionEvent(
            cluster_ids=cluster_ids,
            concept_name=concept_name,
            confidence=confidence,
            entity_id=entity_id,
            centroid_embedding=centroid.tolist(),
        )

    def get_replay_history(
        self,
        limit: int = 100
    ) -> list[ReplayEvent]:
        """Get recent replay history."""
        return self._replay_history[-limit:]

    def get_abstraction_history(
        self,
        limit: int = 100
    ) -> list[AbstractionEvent]:
        """Get recent abstraction history."""
        return self._abstraction_history[-limit:]

    def get_last_cycle_result(self) -> SleepCycleResult | None:
        """Get result of last sleep cycle."""
        return self._last_cycle_result

    def get_stats(self) -> dict:
        """
        Get sleep consolidation statistics.

        Returns:
            Dict with replay and abstraction counts
        """
        stats = {
            "current_phase": self.current_phase.value,
            "total_replays": len(self._replay_history),
            "total_abstractions": len(self._abstraction_history),
            "sharp_wave_ripples": self.swr.get_stats(),
            "homeostatic_enabled": self._homeostatic is not None,
            # P3.4: Interleaved replay stats
            "interleave_enabled": self.interleave_enabled,
            "recent_ratio": self.recent_ratio,
            "replay_batch_size": self.replay_batch_size,
            "last_cycle": {
                "replays": self._last_cycle_result.nrem_replays if self._last_cycle_result else 0,
                "abstractions": self._last_cycle_result.rem_abstractions if self._last_cycle_result else 0,
                "pruned": self._last_cycle_result.pruned_connections if self._last_cycle_result else 0,
                "duration_seconds": self._last_cycle_result.total_duration_seconds if self._last_cycle_result else 0
            }
        }

        # P1D: Add VTA statistics if available
        if self._vta_circuit is not None:
            try:
                stats["vta_circuit"] = self._vta_circuit.get_stats()
            except Exception:
                pass

        return stats

    def clear_history(self) -> None:
        """Clear replay and abstraction history."""
        self._replay_history.clear()
        self._abstraction_history.clear()
        self._last_cycle_result = None


# Convenience function for quick consolidation
async def run_sleep_cycle(
    episodic_memory: EpisodicMemory,
    semantic_memory: SemanticMemory,
    graph_store: GraphStore,
    session_id: str
) -> SleepCycleResult:
    """
    Run a complete sleep consolidation cycle.

    Args:
        episodic_memory: Episodic memory service
        semantic_memory: Semantic memory service
        graph_store: Graph store
        session_id: Session identifier

    Returns:
        SleepCycleResult
    """
    consolidator = SleepConsolidation(
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        graph_store=graph_store
    )
    return await consolidator.full_sleep_cycle(session_id)
