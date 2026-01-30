"""
Episodic memory learning operations (reconsolidation, three-factor learning).

Handles memory reconsolidation, outcome-based learning, and weight updates.
"""

import logging
from uuid import UUID

import numpy as np
from opentelemetry.trace import SpanKind

from ww.observability.tracing import add_span_attribute, traced

logger = logging.getLogger(__name__)


class EpisodicLearningOps:
    """
    Learning operations for episodic memory.

    Handles reconsolidation, outcome learning, and weight updates across
    multiple learning systems (fusion, reranker, gates, etc.).
    """

    def __init__(
        self,
        vector_store,
        embedding_provider,
        reconsolidation_engine,
        dopamine_system,
        three_factor_rule,
        learned_fusion=None,
        learned_reranker=None,
        learned_gate=None,
        ff_encoder=None,
        capsule_layer=None,
    ):
        self.vector_store = vector_store
        self.embedding = embedding_provider
        self.reconsolidation = reconsolidation_engine
        self.dopamine = dopamine_system
        self.three_factor = three_factor_rule
        self.learned_fusion = learned_fusion
        self.learned_reranker = learned_reranker
        self.learned_gate = learned_gate
        self.ff_encoder = ff_encoder
        self.capsule_layer = capsule_layer

    @traced("episodic.apply_reconsolidation", kind=SpanKind.INTERNAL)
    async def apply_reconsolidation(
        self,
        episode_ids: list[UUID],
        query: str,
        outcome_score: float,
        per_memory_rewards: dict[str, float] | None = None,
        use_dopamine: bool = True,
    ) -> int:
        """
        Apply reconsolidation to retrieved memories based on retrieval outcome.

        Args:
            episode_ids: List of episode UUIDs that were retrieved
            query: The query that retrieved these memories
            outcome_score: Success score [0, 1] (0.5 = neutral)
            per_memory_rewards: Optional per-memory reward overrides
            use_dopamine: Whether to use dopamine RPE modulation

        Returns:
            Number of episodes that were reconsolidated
        """
        if not episode_ids:
            return 0

        try:
            # Generate query embedding
            query_emb = await self.embedding.embed_query(query)
            query_emb_np = np.array(query_emb)

            # Fetch current embeddings for the episodes
            id_strings = [str(eid) for eid in episode_ids]
            results = await self.vector_store.get_with_vectors(
                collection=self.vector_store.episodes_collection,
                ids=id_strings,
            )

            if not results:
                return 0

            # Prepare batch of memories for reconsolidation
            memories = []
            per_memory_importance = {}
            per_memory_lr_modulation = {}

            for id_str, payload, vector in results:
                if vector is not None:
                    mem_uuid = UUID(id_str)
                    memories.append((mem_uuid, np.array(vector)))

                    # Importance = stability * (0.5 + valence)
                    stability = payload.get("stability", 1.0)
                    valence = payload.get("emotional_valence", 0.5)
                    importance = stability * (0.5 + valence)
                    per_memory_importance[id_str] = importance

                    # Dopamine RPE modulation
                    if use_dopamine:
                        actual = per_memory_rewards.get(id_str, outcome_score) if per_memory_rewards else outcome_score
                        rpe = self.dopamine.compute_rpe(mem_uuid, actual)
                        per_memory_lr_modulation[id_str] = rpe.surprise_magnitude
                        self.dopamine.update_expectations(mem_uuid, actual)
                    else:
                        per_memory_lr_modulation[id_str] = 1.0

            # Apply reconsolidation
            updates = self.reconsolidation.batch_reconsolidate(
                memories=memories,
                query_embedding=query_emb_np,
                outcome_score=outcome_score,
                per_memory_rewards=per_memory_rewards,
                per_memory_importance=per_memory_importance,
                per_memory_lr_modulation=per_memory_lr_modulation,
            )

            if not updates:
                return 0

            # Persist updated embeddings
            vector_updates = [
                (str(mem_id), new_emb.tolist())
                for mem_id, new_emb in updates.items()
            ]

            updated_count = await self.vector_store.batch_update_vectors(
                collection=self.vector_store.episodes_collection,
                updates=vector_updates,
            )

            logger.info(
                f"Reconsolidated {updated_count} episodes: "
                f"outcome={outcome_score:.2f}"
            )

            add_span_attribute("reconsolidation.count", updated_count)
            add_span_attribute("reconsolidation.outcome", outcome_score)

            return updated_count

        except Exception as e:
            logger.warning(f"Error in reconsolidation: {e}")
            return 0

    async def update_learned_fusion(
        self,
        query: str,
        episode_ids: list[UUID],
        outcome_score: float,
        neural_scores: dict[str, float] | None = None,
        recency_scores: dict[str, float] | None = None,
    ) -> bool:
        """Update learned fusion weights from retrieval outcome."""
        if not self.learned_fusion or not (neural_scores or recency_scores):
            return False

        try:
            # Compute average component scores
            avg_components = {
                "semantic": 0.0,
                "recency": 0.0,
                "outcome": 0.0,
                "importance": 0.0,
            }

            n_memories = len(episode_ids)
            for id_str in [str(eid) for eid in episode_ids]:
                if neural_scores and id_str in neural_scores:
                    avg_components["semantic"] += neural_scores[id_str]
                if recency_scores and id_str in recency_scores:
                    avg_components["recency"] += recency_scores[id_str]

            if n_memories > 0:
                for k in avg_components:
                    avg_components[k] /= n_memories

            # Get query embedding
            query_emb = await self.embedding.embed_query(query)
            query_emb_np = np.array(query_emb)

            # Update fusion weights
            self.learned_fusion.update(
                query_embedding=query_emb_np,
                component_scores=avg_components,
                outcome_utility=outcome_score
            )

            logger.debug(
                f"Updated learned fusion: n_updates={self.learned_fusion.n_updates}"
            )
            return True

        except Exception as e:
            logger.warning(f"Learned fusion update failed: {e}")
            return False

    async def update_learned_reranker(
        self,
        query: str,
        episode_ids: list[UUID],
        outcome_score: float,
        neural_scores: dict[str, float] | None = None,
        recency_scores: dict[str, float] | None = None,
        per_memory_rewards: dict[str, float] | None = None,
        combined_signals: dict[str, float] | None = None,
    ) -> bool:
        """Update learned reranker from retrieval outcome."""
        if not self.learned_reranker or not (neural_scores or recency_scores):
            return False

        try:
            component_scores_list = []
            outcome_utilities = []

            for id_str in [str(eid) for eid in episode_ids]:
                components = {
                    "semantic": neural_scores.get(id_str, 0.5) if neural_scores else 0.5,
                    "recency": recency_scores.get(id_str, 0.5) if recency_scores else 0.5,
                    "outcome": 0.5,
                    "importance": 0.5,
                }
                component_scores_list.append(components)

                # Determine utility
                if per_memory_rewards and id_str in per_memory_rewards:
                    utility = per_memory_rewards[id_str]
                elif combined_signals and id_str in combined_signals:
                    utility = 0.5 + 0.5 * np.clip(combined_signals[id_str], -1.0, 1.0)
                else:
                    utility = outcome_score
                outcome_utilities.append(utility)

            # Get query embedding
            query_emb = await self.embedding.embed_query(query)
            query_emb_np = np.array(query_emb)

            # Update reranker
            self.learned_reranker.update(
                query_embedding=query_emb_np,
                component_scores_list=component_scores_list,
                outcome_utilities=outcome_utilities,
            )

            logger.debug(
                f"Updated learned reranker: n_updates={self.learned_reranker.n_updates}"
            )
            return True

        except Exception as e:
            logger.warning(f"Learned reranker update failed: {e}")
            return False
