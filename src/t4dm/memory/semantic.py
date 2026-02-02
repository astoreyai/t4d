"""
Semantic Memory Service for T4DM.

Implements Hebbian-weighted knowledge graph with ACT-R activation retrieval.
"""

import asyncio
import logging
import math
import random
from collections import OrderedDict
from datetime import datetime
from time import time

# Import type for optional plasticity integration
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from opentelemetry.trace import SpanKind

from t4dm.core.config import get_settings
from t4dm.core.types import Entity, EntityType, Relationship, RelationType, ScoredResult
from t4dm.embedding.bge_m3 import get_embedding_provider
from t4dm.observability.tracing import add_span_attribute, traced
from t4dm.storage import get_graph_store, get_vector_store

if TYPE_CHECKING:
    from t4dm.learning.plasticity import PlasticityManager

# P7.1: Bridge container for NCA subsystem integration
from t4dm.core.bridge_container import get_bridge_container

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Semantic memory service.

    Stores abstracted knowledge with Hebbian-weighted relationships.
    Implements ACT-R activation-based retrieval with spreading activation.
    """

    def __init__(
        self,
        session_id: str | None = None,
        plasticity_manager: Optional["PlasticityManager"] = None,
    ):
        """
        Initialize semantic memory service.

        Args:
            session_id: Session identifier for instance isolation
            plasticity_manager: Optional PlasticityManager for LTD/homeostatic scaling
        """
        settings = get_settings()
        self.session_id = session_id or settings.session_id

        self.embedding = get_embedding_provider()
        self.vector_store = get_vector_store(self.session_id)
        self.graph_store = get_graph_store(self.session_id)

        # Plasticity integration (LTD, homeostatic scaling, metaplasticity)
        self.plasticity_manager = plasticity_manager

        # Hebbian parameters
        self.learning_rate = settings.hebbian_learning_rate
        self.initial_weight = settings.hebbian_initial_weight
        self.decay_rate = settings.hebbian_decay_rate
        self.min_weight = settings.hebbian_min_weight
        self.stale_days = settings.hebbian_stale_days

        # ACT-R parameters
        self.decay = settings.actr_decay
        self.threshold = settings.actr_threshold
        self.noise = settings.actr_noise
        self.spreading_strength = settings.actr_spreading_strength

        # FSRS parameters
        self.fsrs_decay_factor = settings.fsrs_decay_factor

        # Retrieval weights
        self.semantic_weight = settings.semantic_weight_similarity
        self.activation_weight = settings.semantic_weight_activation
        self.retrievability_weight = settings.semantic_weight_retrievability

        # P2-OPT-B4.2.1: LRU cache for entity relationship lookups
        # Cache: key -> (timestamp, strength, fan_out)
        self._entity_cache: OrderedDict[str, tuple[float, float, int]] = OrderedDict()
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300.0  # 5 minutes

        # P7.1: Bridge container for Capsule retrieval bridge integration
        # Capsule bridge provides part-whole scoring to boost semantic retrieval
        self._bridge_container = get_bridge_container(self.session_id)
        self._capsule_retrieval_enabled = getattr(settings, "capsule_retrieval_enabled", True)

    async def initialize(self) -> None:
        """Initialize storage backends."""
        await self.vector_store.initialize()
        await self.graph_store.initialize()

    @traced("semantic.create_entity", kind=SpanKind.INTERNAL)
    async def create_entity(
        self,
        name: str,
        entity_type: str,
        summary: str,
        details: str | None = None,
        source: str | None = None,
    ) -> Entity:
        """
        Create a semantic knowledge entity.

        Args:
            name: Canonical entity name
            entity_type: CONCEPT, PERSON, PROJECT, TOOL, TECHNIQUE, FACT
            summary: Short description
            details: Expanded context
            source: episode_id or 'user_provided'

        Returns:
            Created entity with embedding
        """
        # Generate embedding from name + summary
        embed_text = f"{name}: {summary}"
        embedding = await self.embedding.embed_query(embed_text)

        # Create entity object
        entity = Entity(
            name=name,
            entity_type=EntityType(entity_type),
            summary=summary,
            details=details,
            embedding=embedding,
            source=source,
        )

        # Store in T4DX (single engine handles both vector and graph)
        await self.vector_store.add(
            collection=self.vector_store.entities_collection,
            ids=[str(entity.id)],
            vectors=[embedding],
            payloads=[self._to_payload(entity)],
        )
        await self.graph_store.create_node(
            label="Entity",
            properties=self._to_graph_props(entity),
        )

        logger.info(f"Created entity '{name}' ({entity_type})")
        return entity

    async def create_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: str,
        initial_weight: float | None = None,
    ) -> Relationship:
        """
        Create a Hebbian-weighted relationship between entities.

        Args:
            source_id: Source entity UUID
            target_id: Target entity UUID
            relation_type: USES, PRODUCES, REQUIRES, CAUSES, etc.
            initial_weight: Starting weight (default: 0.1)

        Returns:
            Created relationship
        """
        weight = initial_weight if initial_weight is not None else self.initial_weight

        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType(relation_type),
            weight=weight,
        )

        # Store in graph database
        await self.graph_store.create_relationship(
            source_id=str(source_id),
            target_id=str(target_id),
            rel_type=relation_type,
            properties={
                "weight": weight,
                "coAccessCount": 1,
                "lastCoAccess": datetime.now().isoformat(),
            },
        )

        logger.info(f"Created relationship {source_id} -{relation_type}-> {target_id}")
        return relationship

    @traced("semantic.recall", kind=SpanKind.INTERNAL)
    async def recall(
        self,
        query: str,
        context_entities: list[str] | None = None,
        limit: int = 10,
        include_spreading: bool = True,
        session_filter: str | None = None,
    ) -> list[ScoredResult]:
        """
        Retrieve entities with ACT-R activation scoring.

        Combines:
        1. Vector similarity (semantic match)
        2. ACT-R activation (recency/frequency + spreading)
        3. FSRS retrievability (decay)

        Args:
            query: Natural language search query
            context_entities: Entity IDs for spreading activation
            limit: Maximum results
            include_spreading: Include spreading activation from context
            session_filter: Filter to specific session (defaults to current)

        Returns:
            Scored entities with component breakdown
        """
        # Generate query embedding
        query_vec = await self.embedding.embed_query(query)

        # Build session filter
        filter_dict = {}
        if session_filter:
            filter_dict["session_id"] = session_filter
        elif self.session_id != "default":
            filter_dict["session_id"] = self.session_id

        # Vector search for candidates
        results = await self.vector_store.search(
            collection=self.vector_store.entities_collection,
            vector=query_vec,
            limit=limit * 3,
            filter=filter_dict if filter_dict else None,
        )

        # Load context entities for spreading activation
        context = []
        if include_spreading and context_entities:
            for eid in context_entities:
                entity = await self.get_entity(UUID(eid))
                if entity:
                    context.append(entity)

        # Pre-load context relationships for performance
        context_cache = None
        if context:
            context_cache = await self._preload_context_relationships(context)

        current_time = datetime.now()
        scored_results = []

        for id_str, similarity, payload in results:
            entity = self._from_payload(id_str, payload)

            # Skip invalid entities
            if not entity.is_valid(current_time):
                continue

            # Calculate ACT-R activation (uses cached context data)
            activation = await self._calculate_activation(
                entity,
                context,
                current_time,
                context_cache,
            )

            # Calculate FSRS-4.5 retrievability using correct exponential formula
            # R = 0.9^(t/S) = exp(ln(0.9) * t / S)
            # where t=elapsed_days, S=stability (time for R to fall to 90%)
            elapsed_days = (current_time - entity.last_accessed).total_seconds() / 86400
            # CRASH-001 FIX: Ensure stability is positive to prevent division by zero
            stability = max(entity.stability, 1e-6)
            if elapsed_days <= 0:
                retrievability = 1.0
            else:
                # FSRS-4.5 formula: R = exp(ln(0.9) * t / S)
                retrievability = math.exp(math.log(0.9) * elapsed_days / stability)

            # Sigmoid to normalize activation
            norm_activation = 1 / (1 + math.exp(-activation))

            # Combined score
            total_score = (
                self.semantic_weight * similarity +
                self.activation_weight * norm_activation +
                self.retrievability_weight * retrievability
            )

            scored_results.append(ScoredResult(
                item=entity,
                score=total_score,
                components={
                    "semantic": similarity,
                    "activation": norm_activation,
                    "retrievability": retrievability,
                },
            ))

        # Sort by score and limit
        scored_results.sort(key=lambda x: x.score, reverse=True)
        results = scored_results[:limit]

        # P7.1: Capsule Retrieval Bridge for part-whole agreement scoring
        # Uses capsule network representations to boost semantically aligned memories
        if self._capsule_retrieval_enabled and results:
            try:
                capsule_bridge = self._bridge_container.get_capsule_bridge()
                if capsule_bridge is not None:
                    # Get embeddings for capsule agreement computation
                    # Note: For efficiency, we only compute boosts for top candidates
                    import numpy as np
                    query_emb = np.array(query_vec)

                    # Extract embeddings from candidate entities
                    candidate_embeddings = []
                    for scored in results:
                        if hasattr(scored.item, 'embedding') and scored.item.embedding is not None:
                            candidate_embeddings.append(np.array(scored.item.embedding))
                        else:
                            # Generate embedding if not available
                            entity_text = f"{scored.item.name}: {scored.item.summary}"
                            emb = await self.embedding.embed_query(entity_text)
                            candidate_embeddings.append(np.array(emb))

                    # Compute capsule agreement boosts
                    if candidate_embeddings:
                        boosts = capsule_bridge.compute_boosts(query_emb, candidate_embeddings)

                        # Apply boosts to scores
                        boosted_results = []
                        for i, scored in enumerate(results):
                            boost = boosts[i] if i < len(boosts) else 0.0
                            new_score = scored.score + boost

                            # Track capsule boost in components
                            components = dict(scored.components) if scored.components else {}
                            components["capsule_boost"] = boost

                            boosted_results.append(ScoredResult(
                                item=scored.item,
                                score=new_score,
                                components=components,
                            ))

                        # Re-sort by boosted scores
                        boosted_results.sort(key=lambda x: x.score, reverse=True)
                        results = boosted_results

                        logger.debug(
                            f"P7.1: Capsule boosts applied - "
                            f"mean_boost={sum(boosts)/len(boosts):.4f}"
                        )

            except Exception as e:
                logger.warning(f"Capsule retrieval bridge failed: {e}")

        # Hebbian strengthening for co-retrieved entities
        await self._strengthen_co_retrieval(results)

        # Plasticity integration: LTD and homeostatic scaling
        # This applies long-term depression to non-co-activated neighbors
        # and maintains network stability through homeostatic scaling
        if self.plasticity_manager and results:
            activated_ids = {str(r.item.id) for r in results}
            try:
                await self.plasticity_manager.on_retrieval(
                    activated_ids=activated_ids,
                    store=self.graph_store,
                )
            except Exception as e:
                logger.warning(f"Plasticity on_retrieval failed: {e}")

        return results

    async def _calculate_activation(
        self,
        entity: Entity,
        context: list[Entity],
        current_time: datetime,
        context_cache: dict | None = None,
    ) -> float:
        """
        Calculate ACT-R total activation.

        A = B + sum(W * S) + noise

        Args:
            entity: Entity to calculate activation for
            context: Context entities for spreading activation
            current_time: Current timestamp
            context_cache: Pre-loaded context relationships for optimization
        """
        # Base-level activation from access history (ACT-R formula)
        # B = ln(access_count) - d * ln(time_since_access_in_hours)
        # Use log scale because cognitive activation follows power law (not linear)
        # Decay parameter d=0.5 (default) balances recency vs frequency
        elapsed = (current_time - entity.last_accessed).total_seconds()
        # CRASH-004 FIX: Ensure access_count >= 1 to prevent math.log(0) domain error
        access_count = max(entity.access_count, 1)
        if elapsed > 0:
            base = math.log(access_count) - self.decay * math.log(elapsed / 3600)
        else:
            base = math.log(access_count)

        # Spreading activation from context
        spreading = 0.0
        if context:
            # W = attention weight (inverse of context size - limited attention divided)
            W = 1.0 / len(context)
            # S = strength parameter (configured, ACT-R default is 1.6 for semantic associations)
            S = self.spreading_strength

            for src in context:
                # Use cached data if available
                if context_cache and str(src.id) in context_cache:
                    cache = context_cache[str(src.id)]
                    strength = cache["strengths"].get(str(entity.id), 0.0)
                    fan = cache["fan_out"]
                else:
                    # Fallback to individual queries
                    strength = await self._get_connection_strength(src.id, entity.id)
                    fan = await self._get_fan_out(src.id) if strength > 0 else 1

                if strength > 0:
                    # Fan effect: entities with many connections spread less activation
                    # (cognitive principle: specific concepts activate neighbors more than broad ones)
                    # Formula: W * Hebbian_weight * (S - ln(fan_out))
                    spreading += W * strength * (S - math.log(max(fan, 1)))

        # Add noise
        noise = random.gauss(0, self.noise)

        return base + spreading + noise

    async def _preload_context_relationships(
        self,
        context: list[Entity],
    ) -> dict:
        """
        Pre-load relationships for all context entities using batch query.

        Returns cache dict: {entity_id: {"strengths": {target_id: weight}, "fan_out": int}}
        """
        if not context:
            return {}

        entity_ids = [str(e.id) for e in context]

        # Batch fetch all relationships (both directions) in single query
        both_rels = await self.graph_store.get_relationships_batch(
            node_ids=entity_ids,
            direction="both",
        )

        # Batch fetch outgoing relationships for fan-out calculation
        out_rels = await self.graph_store.get_relationships_batch(
            node_ids=entity_ids,
            direction="out",
        )

        # Build cache
        cache = {}
        for entity_id in entity_ids:
            strengths = {}
            for rel in both_rels.get(entity_id, []):
                strengths[rel["other_id"]] = rel["properties"].get("weight", 0.0)

            cache[entity_id] = {
                "strengths": strengths,
                "fan_out": len(out_rels.get(entity_id, [])),
            }

        return cache

    # P2-OPT-B4.2.1: LRU cache helpers for entity relationship lookups

    def _cache_key(self, entity_id: UUID) -> str:
        """Generate cache key for entity."""
        return str(entity_id)

    def _cache_get(self, key: str) -> tuple[dict[str, float], int] | None:
        """Get cached strengths and fan_out for entity. Returns None if miss/expired."""
        if key in self._entity_cache:
            ts, strengths, fan_out = self._entity_cache[key]
            if time() - ts < self._cache_ttl_seconds:
                self._entity_cache.move_to_end(key)  # LRU: move to end
                return strengths, fan_out
            del self._entity_cache[key]  # Expired
        return None

    def _cache_set(self, key: str, strengths: dict[str, float], fan_out: int) -> None:
        """Cache strengths and fan_out for entity with LRU eviction."""
        self._entity_cache[key] = (time(), strengths, fan_out)
        self._entity_cache.move_to_end(key)
        # Evict oldest entries if over capacity
        while len(self._entity_cache) > self._cache_max_size:
            self._entity_cache.popitem(last=False)

    def clear_entity_cache(self) -> None:
        """Clear the entity relationship cache."""
        self._entity_cache.clear()

    async def _get_connection_strength(
        self,
        source_id: UUID,
        target_id: UUID,
    ) -> float:
        """Get Hebbian weight between entities (with LRU cache)."""
        # P2-OPT-B4.2.1: Check cache first
        cache_key = self._cache_key(source_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            strengths, _ = cached
            return strengths.get(str(target_id), 0.0)

        # Cache miss - fetch and cache
        rels = await self.graph_store.get_relationships(
            node_id=str(source_id),
            direction="both",
        )

        # Build strengths dict for cache
        strengths = {}
        for rel in rels:
            strengths[rel["other_id"]] = rel["properties"].get("weight", 0.0)

        # Also get fan_out for this entity
        out_rels = await self.graph_store.get_relationships(
            node_id=str(source_id),
            direction="out",
        )
        fan_out = len(out_rels)

        # Cache the result
        self._cache_set(cache_key, strengths, fan_out)

        return strengths.get(str(target_id), 0.0)

    async def _get_fan_out(self, entity_id: UUID) -> int:
        """Get number of outgoing connections (with LRU cache)."""
        # P2-OPT-B4.2.1: Check cache first
        cache_key = self._cache_key(entity_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            _, fan_out = cached
            return fan_out

        # Cache miss - fetch and cache
        rels = await self.graph_store.get_relationships(
            node_id=str(entity_id),
            direction="out",
        )
        fan_out = len(rels)

        # Also fetch strengths for complete cache entry
        both_rels = await self.graph_store.get_relationships(
            node_id=str(entity_id),
            direction="both",
        )
        strengths = {}
        for rel in both_rels:
            strengths[rel["other_id"]] = rel["properties"].get("weight", 0.0)

        # Cache the result
        self._cache_set(cache_key, strengths, fan_out)

        return fan_out

    async def _strengthen_co_retrieval(self, results: list[ScoredResult]) -> None:
        """Apply Hebbian strengthening to co-retrieved entities.

        Optimized with batch relationship queries and parallel strengthening.
        Eliminates N+1 query pattern.
        """
        if len(results) < 2:
            return

        entities = [r.item for r in results]
        entity_ids = [str(e.id) for e in entities]

        # Batch fetch all relationships in single query
        try:
            relationships_map = await self.graph_store.get_relationships_batch(
                node_ids=entity_ids,
                direction="both",
            )
        except Exception as e:
            logger.error(f"Failed to batch fetch relationships: {e}")
            return

        # Build connection strength lookup
        strength_lookup = {}
        for node_id, rels in relationships_map.items():
            for rel in rels:
                other_id = rel["other_id"]
                weight = rel["properties"].get("weight", 0.0)
                # Store bidirectional lookup
                key1 = (node_id, other_id)
                key2 = (other_id, node_id)
                strength_lookup[key1] = weight
                strength_lookup[key2] = weight

        # Build pairs to strengthen (only those with existing relationships)
        pairs_to_strengthen = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                key = (str(e1.id), str(e2.id))
                if key in strength_lookup and strength_lookup[key] > 0:
                    pairs_to_strengthen.append((e1, e2))

        if not pairs_to_strengthen:
            return

        # Parallel Hebbian strengthening
        async def strengthen_pair(e1, e2):
            try:
                await self.graph_store.strengthen_relationship(
                    source_id=str(e1.id),
                    target_id=str(e2.id),
                    learning_rate=self.learning_rate,
                )
            except Exception as e:
                logger.warning(f"Failed to strengthen {e1.id}-{e2.id}: {e}")

        # Execute all strengthening in parallel
        await asyncio.gather(*[
            strengthen_pair(e1, e2)
            for e1, e2 in pairs_to_strengthen
        ], return_exceptions=True)

        logger.debug(f"Strengthened {len(pairs_to_strengthen)} co-retrieval relationships")

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get entity by ID."""
        results = await self.vector_store.get(
            collection=self.vector_store.entities_collection,
            ids=[str(entity_id)],
        )

        if results:
            id_str, payload = results[0]
            return self._from_payload(id_str, payload)

        return None

    async def list_entities(
        self,
        limit: int = 50,
        session_filter: str | None = None,
    ) -> list[Entity]:
        """
        List entities with optional session filtering.

        Args:
            limit: Maximum number of entities to return (default 50, max 500)
            session_filter: Optional session ID filter

        Returns:
            List of entities
        """
        limit = min(limit, 500)

        # Build filter for current session
        filter_conditions = {}
        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        try:
            results, _ = await self.vector_store.scroll(
                collection=self.vector_store.entities_collection,
                scroll_filter=filter_conditions if filter_conditions else None,
                limit=limit,
                offset=0,
                with_payload=True,
                with_vectors=False,
            )

            entities = []
            for id_str, payload, _ in results:
                entity = self._from_payload(id_str, payload)
                entities.append(entity)

            logger.debug(f"Listed {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Error listing entities: {e}")
            raise

    async def get_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 50,
        session_filter: str | None = None,
    ) -> list[Entity]:
        """
        Get entities filtered by type.

        Args:
            entity_type: Type to filter by
            limit: Maximum number of entities to return
            session_filter: Optional session ID filter

        Returns:
            List of entities of the specified type
        """
        limit = min(limit, 500)

        # Build filter
        filter_conditions = {"entity_type": entity_type.value}
        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        try:
            results, _ = await self.vector_store.scroll(
                collection=self.vector_store.entities_collection,
                scroll_filter=filter_conditions,
                limit=limit,
                offset=0,
                with_payload=True,
                with_vectors=False,
            )

            entities = []
            for id_str, payload, _ in results:
                entity = self._from_payload(id_str, payload)
                entities.append(entity)

            logger.debug(f"Found {len(entities)} entities of type {entity_type}")
            return entities

        except Exception as e:
            logger.error(f"Error getting entities by type: {e}")
            raise

    async def supersede(
        self,
        entity_id: UUID,
        new_summary: str,
        new_details: str | None = None,
    ) -> Entity:
        """
        Update entity with bi-temporal versioning.

        Old version's validTo is set, new version created.

        Args:
            entity_id: Entity to supersede
            new_summary: Updated summary
            new_details: Updated details

        Returns:
            New entity version
        """
        old = await self.get_entity(entity_id)
        if not old:
            raise ValueError(f"Entity {entity_id} not found")

        # Store current time for consistency
        now = datetime.now()
        now_iso = now.isoformat()

        # Close validity in T4DX (single engine)
        await self.graph_store.update_node(
            node_id=str(entity_id),
            properties={"validTo": now_iso},
            label="Entity",
        )

        logger.debug(f"Closed validity for entity {entity_id}")

        # Create new version
        return await self.create_entity(
            name=old.name,
            entity_type=old.entity_type.value,
            summary=new_summary,
            details=new_details or old.details,
            source=old.source,
        )

    async def spread_activation(
        self,
        seed_entities: list[str],
        steps: int = 3,
        retention: float = 0.5,
        decay: float = 0.1,
        threshold: float = 0.01,
        max_nodes: int = 1000,
        max_neighbors_per_node: int = 50,
        session_id: str | None = None,
    ) -> dict[str, float]:
        """
        Spread activation through knowledge graph with explosion prevention.

        Implements ACT-R spreading activation with safeguards against
        unbounded graph traversal.

        Args:
            seed_entities: Entity IDs to start activation from
            steps: Number of propagation steps (1-5)
            retention: Fraction of activation retained at each node (0-1)
            decay: Activation decay per step (0-1)
            threshold: Minimum activation to continue spreading
            max_nodes: Maximum total nodes in activation map (prevents explosion)
            max_neighbors_per_node: Maximum neighbors to consider per node
            session_id: Optional session filter

        Returns:
            Dict mapping entity_id -> activation level

        Example:
            >>> activation = await semantic.spread_activation(
            ...     seed_entities=["entity-123", "entity-456"],
            ...     steps=2,
            ...     max_nodes=500,
            ... )
            >>> # Returns {"entity-123": 1.0, "entity-456": 1.0, "related-1": 0.4, ...}
        """
        # Validate parameters
        steps = min(max(1, steps), 5)  # Clamp to 1-5
        retention = min(max(0.0, retention), 1.0)
        decay = min(max(0.0, decay), 1.0)
        max_nodes = min(max(10, max_nodes), 10000)  # Clamp to 10-10000
        max_neighbors_per_node = min(max(1, max_neighbors_per_node), 200)

        if not seed_entities:
            return {}

        # Initialize activation map with seeds
        activation: dict[str, float] = dict.fromkeys(seed_entities, 1.0)
        visited: set[str] = set(seed_entities)

        logger.debug(
            f"Starting spread activation: {len(seed_entities)} seeds, "
            f"{steps} steps, max_nodes={max_nodes}"
        )

        for step in range(steps):
            if len(activation) >= max_nodes:
                logger.warning(
                    f"Spread activation hit max_nodes limit ({max_nodes}) at step {step}"
                )
                break

            new_activation: dict[str, float] = {}
            decay_factor = 1.0 - (decay * step)  # Progressive decay

            # Process nodes in order of activation (highest first)
            sorted_nodes = sorted(
                activation.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:max_nodes]  # Only process top nodes

            for entity_id, act in sorted_nodes:
                if act < threshold:
                    continue

                # Retain some activation at current node
                retained = act * retention * decay_factor
                new_activation[entity_id] = new_activation.get(entity_id, 0) + retained

                try:
                    # Get weighted neighbors (limited)
                    neighbors = await self.graph_store.get_relationships(
                        node_id=entity_id,
                        direction="both",
                        limit=max_neighbors_per_node,
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to get neighbors for {entity_id}: {e}")
                    continue

                if not neighbors:
                    continue

                # Calculate spread amount
                spread_amount = act * (1 - retention) * decay_factor

                # Filter neighbors: prefer unvisited, high-weight connections
                valid_neighbors = []
                for neighbor in neighbors:
                    neighbor_id = neighbor.get("other_id")
                    if neighbor_id == entity_id:
                        continue

                    weight = neighbor.get("properties", {}).get("weight", 1.0)
                    # Boost unvisited nodes
                    novelty_bonus = 1.5 if neighbor_id not in visited else 1.0
                    score = weight * novelty_bonus
                    valid_neighbors.append((neighbor_id, weight, score))

                if not valid_neighbors:
                    continue

                # Sort by score and take top neighbors
                valid_neighbors.sort(key=lambda x: x[2], reverse=True)
                valid_neighbors = valid_neighbors[:max_neighbors_per_node]

                # Normalize weights for distribution
                total_weight = sum(w for _, w, _ in valid_neighbors)
                if total_weight == 0:
                    continue

                # Spread activation to neighbors
                for neighbor_id, weight, _ in valid_neighbors:
                    # Check max_nodes limit
                    if (
                        neighbor_id not in new_activation
                        and len(new_activation) >= max_nodes
                    ):
                        continue

                    # Calculate neighbor's activation
                    normalized_weight = weight / total_weight
                    neighbor_activation = spread_amount * normalized_weight

                    if neighbor_activation >= threshold:
                        new_activation[neighbor_id] = (
                            new_activation.get(neighbor_id, 0) + neighbor_activation
                        )
                        visited.add(neighbor_id)

            # Update activation map
            activation = new_activation

            logger.debug(
                f"Step {step + 1}: {len(activation)} nodes, "
                f"max_activation={max(activation.values()) if activation else 0:.3f}"
            )

        # Filter by threshold and return
        result = {
            eid: act for eid, act in activation.items()
            if act >= threshold
        }

        logger.info(
            f"Spread activation complete: {len(result)} nodes from {len(seed_entities)} seeds"
        )

        return result

    @traced("semantic.apply_hebbian_decay", kind=SpanKind.INTERNAL)
    async def apply_hebbian_decay(
        self,
        decay_rate: float | None = None,
        min_weight: float | None = None,
        stale_days: int | None = None,
        session_id: str | None = None,
    ) -> dict:
        """
        Apply decay to relationships not accessed recently.

        Uses batch operations for efficiency - O(1) database queries
        instead of O(n) individual updates.

        Hebbian decay formula: w' = w * (1 - decay_rate)

        This prevents unbounded growth and removes stale relationships.

        Args:
            decay_rate: Rate of decay per application (default from config)
            min_weight: Minimum weight before pruning (default from config)
            stale_days: Days since last access to consider stale (default from config)
            session_id: Optional session filter

        Returns:
            dict with decayed_count, pruned_count, total_processed
        """
        get_settings()

        # Use configured defaults if not specified
        decay_rate = decay_rate if decay_rate is not None else self.decay_rate
        min_weight = min_weight if min_weight is not None else self.min_weight
        stale_days = stale_days if stale_days is not None else self.stale_days

        # Get count for logging
        stale_count = await self.graph_store.count_stale_relationships(
            stale_days=stale_days,
            session_id=session_id or self.session_id,
        )

        if stale_count == 0:
            logger.debug("No stale relationships to decay")
            return {
                "decayed_count": 0,
                "pruned_count": 0,
                "total_processed": 0,
                "decay_rate": decay_rate,
                "min_weight": min_weight,
                "stale_days": stale_days,
            }

        logger.info(f"Applying Hebbian decay to {stale_count} stale relationships")

        # Use batch operation
        result = await self.graph_store.batch_decay_relationships(
            stale_days=stale_days,
            decay_rate=decay_rate,
            min_weight=min_weight,
            session_id=session_id or self.session_id,
        )

        decayed = result["decayed"]
        pruned = result["pruned"]

        logger.info(
            f"Hebbian decay: decayed={decayed}, pruned={pruned}, "
            f"total_processed={stale_count} "
            f"(decay_rate={decay_rate}, min_weight={min_weight}, stale_days={stale_days})"
        )

        add_span_attribute("decay.decayed_count", decayed)
        add_span_attribute("decay.pruned_count", pruned)
        add_span_attribute("decay.total_processed", stale_count)

        return {
            "decayed_count": decayed,
            "pruned_count": pruned,
            "total_processed": stale_count,
            "decay_rate": decay_rate,
            "min_weight": min_weight,
            "stale_days": stale_days,
        }

    @traced("semantic.learn_from_outcome", kind=SpanKind.INTERNAL)
    async def learn_from_outcome(
        self,
        entity_ids: list[UUID],
        outcome_score: float,
        query: str | None = None,
    ) -> dict:
        """
        Apply outcome-modulated Hebbian learning to entity relationships.

        PHASE-2 LEARNING WIRING: This method modulates Hebbian strengthening
        based on retrieval outcomes:
        - Positive outcome (>0.5): Strengthen relationships more than baseline
        - Negative outcome (<0.5): Weaken relationships (anti-Hebbian)
        - Neutral outcome (=0.5): No change

        This implements reward-modulated Hebbian learning:
            Δw = learning_rate × (outcome - 0.5) × 2

        For positive outcomes: Δw is positive → strengthen
        For negative outcomes: Δw is negative → weaken

        Args:
            entity_ids: List of entity UUIDs that were retrieved
            outcome_score: Overall success score [0, 1]
            query: Optional query string (for future use in query-dependent learning)

        Returns:
            Dict with strengthened_count, weakened_count, avg_delta
        """
        if len(entity_ids) < 2:
            return {
                "strengthened_count": 0,
                "weakened_count": 0,
                "avg_delta": 0.0,
            }

        # Compute advantage (centered outcome)
        advantage = (outcome_score - 0.5) * 2  # Scale to [-1, 1]

        # Skip neutral outcomes
        if abs(advantage) < 0.02:
            logger.debug("Neutral outcome, skipping Hebbian update")
            return {
                "strengthened_count": 0,
                "weakened_count": 0,
                "avg_delta": 0.0,
            }

        entity_id_strs = [str(eid) for eid in entity_ids]
        strengthened = 0
        weakened = 0
        total_delta = 0.0

        # Get existing relationships between entities
        try:
            relationships_map = await self.graph_store.get_relationships_batch(
                node_ids=entity_id_strs,
                direction="both",
            )
        except Exception as e:
            logger.warning(f"Failed to get relationships for learning: {e}")
            return {
                "strengthened_count": 0,
                "weakened_count": 0,
                "avg_delta": 0.0,
            }

        # Build pairs to update
        updated_pairs = set()
        for node_id in entity_id_strs:
            if node_id not in relationships_map:
                continue
            for rel in relationships_map[node_id]:
                other_id = rel["other_id"]
                if other_id in entity_id_strs:
                    # Normalize pair order to avoid duplicates
                    pair = tuple(sorted([node_id, other_id]))
                    if pair not in updated_pairs:
                        updated_pairs.add(pair)

        # Apply outcome-modulated learning
        modulated_lr = self.learning_rate * advantage

        async def update_pair(pair):
            nonlocal strengthened, weakened, total_delta
            source_id, target_id = pair
            try:
                if advantage > 0:
                    # Strengthen for positive outcomes
                    await self.graph_store.strengthen_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        learning_rate=modulated_lr,
                    )
                    strengthened += 1
                else:
                    # Weaken for negative outcomes (anti-Hebbian)
                    await self.graph_store.weaken_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        decay_rate=abs(modulated_lr),
                    )
                    weakened += 1
                total_delta += modulated_lr
            except Exception as e:
                logger.warning(f"Failed to update relationship {source_id}-{target_id}: {e}")

        # Execute updates in parallel
        await asyncio.gather(*[
            update_pair(pair)
            for pair in updated_pairs
        ], return_exceptions=True)

        n_updated = strengthened + weakened
        avg_delta = total_delta / n_updated if n_updated > 0 else 0.0

        logger.info(
            f"Outcome-modulated Hebbian: outcome={outcome_score:.2f}, "
            f"strengthened={strengthened}, weakened={weakened}, avg_delta={avg_delta:.4f}"
        )

        add_span_attribute("learning.outcome_score", outcome_score)
        add_span_attribute("learning.strengthened", strengthened)
        add_span_attribute("learning.weakened", weakened)

        return {
            "strengthened_count": strengthened,
            "weakened_count": weakened,
            "avg_delta": avg_delta,
        }

    def _to_payload(self, entity: Entity) -> dict:
        """Convert entity to Qdrant payload."""
        return {
            "session_id": self.session_id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "summary": entity.summary,
            "details": entity.details,
            "source": entity.source,
            "stability": entity.stability,
            "access_count": entity.access_count,
            "last_accessed": entity.last_accessed.isoformat(),
            "created_at": entity.created_at.isoformat(),
            "valid_from": entity.valid_from.isoformat(),
            "valid_to": entity.valid_to.isoformat() if entity.valid_to else None,
        }

    def _from_payload(self, id_str: str, payload: dict) -> Entity:
        """Reconstruct entity from Qdrant payload."""
        return Entity(
            id=UUID(id_str),
            name=payload["name"],
            entity_type=EntityType(payload["entity_type"]),
            summary=payload["summary"],
            details=payload.get("details"),
            embedding=None,
            source=payload.get("source"),
            stability=payload["stability"],
            access_count=payload["access_count"],
            last_accessed=datetime.fromisoformat(payload["last_accessed"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            valid_from=datetime.fromisoformat(payload["valid_from"]),
            valid_to=datetime.fromisoformat(payload["valid_to"]) if payload.get("valid_to") else None,
        )

    def _to_graph_props(self, entity: Entity) -> dict:
        """Convert entity to Neo4j properties."""
        return {
            "id": str(entity.id),
            "sessionId": self.session_id,
            "name": entity.name,
            "entityType": entity.entity_type.value,
            "summary": entity.summary,
            "details": entity.details or "",
            "source": entity.source or "",
            "stability": entity.stability,
            "accessCount": entity.access_count,
            "lastAccessed": entity.last_accessed.isoformat(),
            "createdAt": entity.created_at.isoformat(),
            "validFrom": entity.valid_from.isoformat(),
            "validTo": entity.valid_to.isoformat() if entity.valid_to else "",
        }


# Factory function
def get_semantic_memory(
    session_id: str | None = None,
    plasticity_manager: Optional["PlasticityManager"] = None,
) -> SemanticMemory:
    """
    Get semantic memory service instance.

    Args:
        session_id: Session identifier for instance isolation
        plasticity_manager: Optional PlasticityManager for LTD/homeostatic scaling
    """
    return SemanticMemory(session_id, plasticity_manager)
