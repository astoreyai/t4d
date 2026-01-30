# World Weaver Architecture Refactoring Plan

**Version**: 0.5.0 → 0.6.0
**Status**: Design Phase
**Created**: 2026-01-07
**Author**: ww-algorithm (World Weaver Algorithm Design Agent)
**Priority**: P1 (Production Readiness)

---

## Executive Summary

This plan addresses 7 critical architectural issues identified in the [Architecture Review](/mnt/projects/ww/docs/CODE_QUALITY_REVIEW.md) (Score: 7.7/10). The refactoring is structured into 3 phases that can partially overlap with ongoing CompBio and Hinton work.

**Critical Issues**:
1. God Object: `episodic.py` (3,616 lines, 34 methods)
2. Long Method: `create_ww_router()` (429 lines in config.py)
3. 232 print() statements need logger conversion
4. No Redis caching layer
5. No API rate limiting
6. N+1 query problems in graph traversal
7. Low bridge test coverage (5 test files, need 20+)

**Risk Mitigation**: All changes maintain backward compatibility with 8,075 existing tests.

---

## Table of Contents

1. [Phase 1: Episodic Memory Decomposition](#phase-1-episodic-memory-decomposition)
2. [Phase 2: Caching and Performance](#phase-2-caching-and-performance)
3. [Phase 3: Quality and Observability](#phase-3-quality-and-observability)
4. [Testing Strategy](#testing-strategy)
5. [Migration Guide](#migration-guide)
6. [Success Metrics](#success-metrics)

---

## Phase 1: Episodic Memory Decomposition

**Duration**: 2 weeks
**Priority**: P0 (Blocks scaling)
**Dependencies**: None (can start immediately)
**Can overlap with**: CompBio work (different modules)

### Problem Analysis

`/mnt/projects/ww/src/ww/memory/episodic.py` is a 3,616-line God Object with:
- **EpisodicMemory** class (3,139 lines): Storage, retrieval, learning, fusion, reranking
- **LearnedFusionWeights** class (185 lines): Query-dependent scoring
- **LearnedReranker** class (226 lines): Post-retrieval re-ranking
- 34 methods mixing concerns (storage, search, learning, graph, saga)

### Decomposition Strategy

Split into 6 focused modules using **Facade Pattern** to preserve existing API:

```
src/ww/memory/
├── episodic.py              # 400 lines - Facade + backward compat
├── episodic_storage.py      # 800 lines - NEW: Storage operations
├── episodic_retrieval.py    # 1,200 lines - NEW: Search & recall
├── episodic_learning.py     # 600 lines - NEW: Reconsolidation, three-factor
├── episodic_fusion.py       # 400 lines - MOVED: Fusion weights + reranker
└── episodic_saga.py         # 400 lines - NEW: Saga orchestration
```

### Detailed Module Design

#### 1.1 `episodic_storage.py` - Storage Layer

**Responsibility**: CRUD operations, saga coordination, hybrid indexing

```python
"""
Episodic Memory Storage Layer.

Handles create/update/delete operations with saga pattern for dual-store consistency.
"""

from ww.storage.saga import Saga, SagaState
from ww.storage.neo4j_store import Neo4jStore
from ww.storage.qdrant_store import QdrantStore

class EpisodicStorageService:
    """Storage service for episodic memories with saga-based consistency."""

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: Neo4jStore,
        session_id: str,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.session_id = session_id
        self._last_episode_id: UUID | None = None
        self._sequence_counter: int = 0

    async def store(self, episode: Episode, embedding: np.ndarray) -> UUID:
        """
        Store episode with saga pattern.

        Complexity: O(1) storage, O(log n) index update
        """
        ...

    async def store_batch(
        self,
        episodes: list[Episode],
        embeddings: list[np.ndarray]
    ) -> list[UUID]:
        """
        Batch storage for consolidation.

        Complexity: O(k) for k episodes (vs O(k log k) for sequential)
        """
        ...

    async def _store_hybrid(
        self,
        episode: Episode,
        content: str
    ) -> None:
        """Hybrid dense+sparse indexing."""
        ...

    async def _link_episodes(
        self,
        prev_id: UUID,
        next_id: UUID
    ) -> None:
        """Temporal sequencing."""
        ...

    async def update_embedding(
        self,
        episode_id: UUID,
        new_embedding: np.ndarray
    ) -> None:
        """Reconsolidation update."""
        ...

    async def delete(self, episode_id: UUID) -> None:
        """Delete with saga rollback."""
        ...
```

**Lines**: ~800
**Extracted from**: Lines 950-1400, 2300-2500 of episodic.py

---

#### 1.2 `episodic_retrieval.py` - Retrieval Layer

**Responsibility**: Search, scoring, filtering, time-range queries

```python
"""
Episodic Memory Retrieval Layer.

Handles recall operations with FSRS decay, ACT-R activation, and hybrid search.
"""

from ww.memory.cluster_index import ClusterIndex
from ww.memory.learned_sparse_index import LearnedSparseIndex
from ww.embedding.query_memory_separation import QueryMemorySeparator

class EpisodicRetrievalService:
    """Retrieval service with multi-stage search pipeline."""

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: Neo4jStore,
        session_id: str,
        cluster_index: ClusterIndex | None = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.session_id = session_id
        self.cluster_index = cluster_index

        # Retrieval weights
        self.semantic_weight = 0.4
        self.recency_weight = 0.3
        self.outcome_weight = 0.2
        self.importance_weight = 0.1

        # FSRS parameters
        self.default_stability = 1.0
        self.decay_factor = 0.9
        self.recency_decay = 0.95

    async def recall(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        context: EpisodeContext | None = None,
    ) -> list[ScoredResult]:
        """
        Multi-stage recall pipeline.

        Pipeline:
        1. Cluster selection (if enabled): O(log C) for C clusters
        2. Vector search: O(k log n) in selected clusters
        3. Hybrid fusion (if enabled): O(k)
        4. FSRS decay scoring: O(k)
        5. Graph expansion (optional): O(k * d) for depth d

        Total: O(k log n + k*d) amortized
        """
        ...

    async def recall_hybrid(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.7,
    ) -> list[ScoredResult]:
        """
        Hybrid dense + sparse retrieval.

        Complexity: O(k log n) dense + O(k) sparse fusion
        """
        ...

    async def recall_by_timerange(
        self,
        start: datetime,
        end: datetime,
        k: int | None = None,
    ) -> list[ScoredResult]:
        """
        Time-bounded retrieval.

        Complexity: O(m log m) for m items in range
        """
        ...

    def _compute_fsrs_decay(
        self,
        stability: float,
        last_access: datetime
    ) -> float:
        """FSRS decay calculation."""
        ...

    def _compute_activation(
        self,
        episode_dict: dict,
        context: EpisodeContext
    ) -> float:
        """ACT-R activation scoring."""
        ...

    async def _expand_graph_neighbors(
        self,
        episode_ids: list[UUID],
        max_depth: int = 1
    ) -> list[dict]:
        """Graph traversal expansion (uses batch queries)."""
        ...
```

**Lines**: ~1,200
**Extracted from**: Lines 1200-2300 of episodic.py

---

#### 1.3 `episodic_learning.py` - Learning Layer

**Responsibility**: Reconsolidation, pattern separation, dopamine RPE, three-factor learning

```python
"""
Episodic Memory Learning Layer.

Handles memory updating, pattern separation, and credit assignment.
"""

from ww.learning.reconsolidation import ReconsolidationEngine
from ww.learning.three_factor import ThreeFactorLearningRule
from ww.learning.dopamine import DopamineSystem
from ww.memory.pattern_separation import DentateGyrus

class EpisodicLearningService:
    """Learning service for memory plasticity and credit assignment."""

    def __init__(
        self,
        storage_service: EpisodicStorageService,
        embedding_provider,
        vector_store: QdrantStore,
    ):
        self.storage = storage_service
        self.embedding = embedding_provider

        # Three-factor learning rule
        self.three_factor = ThreeFactorLearningRule(
            ach_weight=0.4,
            ne_weight=0.35,
            serotonin_weight=0.25,
        )

        # Reconsolidation engine
        self.reconsolidation = ReconsolidationEngine(
            base_learning_rate=0.1,  # Fast hippocampal learning
            max_update_magnitude=0.2,
            three_factor=self.three_factor,
        )

        # Pattern separation (DG-like)
        self.pattern_separator = DentateGyrus(
            embedding_provider=self.embedding,
            vector_store=vector_store,
            similarity_threshold=0.75,
        )

        # Dopamine reward prediction error
        self.dopamine = DopamineSystem(
            default_expected=0.5,
            value_learning_rate=0.1,
        )

    async def update_from_outcome(
        self,
        episode_id: UUID,
        outcome: Outcome,
        neuromodulator_context: dict[str, float] | None = None,
    ) -> None:
        """
        Update memory based on retrieval outcome.

        Implements:
        1. Dopamine surprise signal: δ = actual - expected
        2. Three-factor learning: LR = f(ACh, NE, 5-HT)
        3. Reconsolidation: embedding ← embedding + α * δ * gradient

        Complexity: O(d) for embedding dimension d
        """
        ...

    async def apply_pattern_separation(
        self,
        content: str,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Apply DG-like pattern separation.

        Complexity: O(k) for k similar memories check
        """
        ...

    async def batch_update_access(
        self,
        episode_ids: list[UUID],
        timestamp: datetime,
    ) -> None:
        """
        Batch update last_accessed for retrieved memories.

        Complexity: O(k) batch update vs O(k log k) sequential
        """
        ...

    def compute_learning_rate(
        self,
        base_lr: float,
        neuromod_context: dict[str, float],
    ) -> float:
        """Three-factor modulated learning rate."""
        ...
```

**Lines**: ~600
**Extracted from**: Lines 2500-3100 of episodic.py

---

#### 1.4 `episodic_fusion.py` - Fusion and Reranking

**Responsibility**: Query-dependent fusion weights, learned reranking

```python
"""
Episodic Memory Fusion and Reranking.

Learned, query-adaptive scoring components.
"""

class LearnedFusionWeights:
    """
    Query-dependent fusion weights for retrieval scoring.

    Replaces fixed weights (0.4/0.3/0.2/0.1) with learned, adaptive weights.
    Uses 2-layer MLP: query_embedding → hidden → softmax(4 weights).

    Complexity:
    - Forward: O(d * h + h * 4) ≈ O(d) for embedding dim d, hidden dim h
    - Update: O(d * h) gradient computation
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
    ):
        # Xavier initialization
        self.W1 = np.random.randn(hidden_dim, embed_dim) * np.sqrt(2.0 / (embed_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(4, hidden_dim) * np.sqrt(2.0 / (hidden_dim + 4))
        self.b2 = np.zeros(4)

    def compute_weights(self, query_embedding: np.ndarray) -> dict[str, float]:
        """Compute query-adaptive weights."""
        ...

    def update(self, query_embedding: np.ndarray, gradient: np.ndarray) -> None:
        """Online gradient descent update."""
        ...


class LearnedReranker:
    """
    Post-retrieval learned re-ranking.

    Uses cross-encoder scoring for top-k refinement.

    Complexity:
    - Rerank: O(k * d^2) for k results, embedding dim d
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 128,
        learning_rate: float = 0.01,
    ):
        ...

    def rerank(
        self,
        query_embedding: np.ndarray,
        results: list[ScoredResult],
    ) -> list[ScoredResult]:
        """Re-rank top results."""
        ...

    def update(
        self,
        query_embedding: np.ndarray,
        result_embedding: np.ndarray,
        relevance_signal: float,
    ) -> None:
        """Update from implicit feedback."""
        ...
```

**Lines**: ~400
**Extracted from**: Lines 64-474 of episodic.py

---

#### 1.5 `episodic_saga.py` - Saga Orchestration

**Responsibility**: Multi-step transaction coordination for dual-store consistency

```python
"""
Episodic Memory Saga Orchestration.

Coordinates multi-step operations across vector and graph stores.
"""

from ww.storage.saga import Saga, SagaState

class EpisodicSagaOrchestrator:
    """Saga coordinator for episodic memory operations."""

    def __init__(
        self,
        storage_service: EpisodicStorageService,
    ):
        self.storage = storage_service

    async def create_episode_saga(
        self,
        episode: Episode,
        embedding: np.ndarray,
    ) -> UUID:
        """
        Create episode with saga pattern.

        Steps:
        1. Add to vector store (Qdrant)
        2. Create node in graph (Neo4j)
        3. Link to previous episode (if exists)

        Rollback on any failure.
        """
        saga = Saga(saga_id=str(uuid4()))

        # Step 1: Vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.storage.vector_store.add(...),
            compensate=lambda: self.storage.vector_store.delete(...),
        )

        # Step 2: Graph node
        saga.add_step(
            name="create_node",
            action=lambda: self.storage.graph_store.create_node(...),
            compensate=lambda: self.storage.graph_store.delete_node(...),
        )

        result = await saga.execute()

        if result.state != SagaState.COMMITTED:
            raise RuntimeError(f"Saga failed: {result.error}")

        return episode.id

    async def update_episode_saga(
        self,
        episode_id: UUID,
        updates: dict[str, Any],
    ) -> None:
        """Update with consistency across stores."""
        ...

    async def delete_episode_saga(
        self,
        episode_id: UUID,
    ) -> None:
        """Delete with cascade and rollback."""
        ...
```

**Lines**: ~400
**Extracted from**: Lines 950-1200 of episodic.py

---

#### 1.6 `episodic.py` - Facade (Refactored)

**Responsibility**: Backward-compatible API, service coordination

```python
"""
Episodic Memory Service for World Weaver.

Facade for episodic memory subsystem with backward compatibility.
"""

from ww.memory.episodic_storage import EpisodicStorageService
from ww.memory.episodic_retrieval import EpisodicRetrievalService
from ww.memory.episodic_learning import EpisodicLearningService
from ww.memory.episodic_fusion import LearnedFusionWeights, LearnedReranker
from ww.memory.episodic_saga import EpisodicSagaOrchestrator

class EpisodicMemory:
    """
    Episodic memory service (Facade).

    Backward-compatible API delegating to specialized services.
    """

    def __init__(self, session_id: str | None = None):
        settings = get_settings()
        self.session_id = session_id or settings.session_id

        # Initialize stores
        self.embedding = get_embedding_provider()
        self.vector_store = get_qdrant_store(self.session_id)
        self.graph_store = get_neo4j_store(self.session_id)

        # Initialize services
        self.storage = EpisodicStorageService(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            session_id=self.session_id,
        )

        self.retrieval = EpisodicRetrievalService(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            session_id=self.session_id,
        )

        self.learning = EpisodicLearningService(
            storage_service=self.storage,
            embedding_provider=self.embedding,
            vector_store=self.vector_store,
        )

        self.saga = EpisodicSagaOrchestrator(
            storage_service=self.storage,
        )

        # Fusion components
        self.fusion_weights = LearnedFusionWeights(
            embed_dim=settings.embedding_dimension,
        )
        self.reranker = LearnedReranker(
            embed_dim=settings.embedding_dimension,
        )

        # Backward compatibility: Expose service attributes
        self.semantic_weight = self.retrieval.semantic_weight
        self.recency_weight = self.retrieval.recency_weight
        self.outcome_weight = self.retrieval.outcome_weight
        self.importance_weight = self.retrieval.importance_weight

    # Delegate to services (backward-compatible API)

    async def store(
        self,
        content: str,
        context: EpisodeContext,
        outcome: Outcome | None = None
    ) -> Episode:
        """Store episode (delegates to storage service)."""
        embedding = await self.embedding.embed(content)
        episode = Episode(...)  # Create episode object
        await self.storage.store(episode, embedding)
        return episode

    async def recall(
        self,
        query: str,
        k: int = 10,
        context: EpisodeContext | None = None,
    ) -> list[ScoredResult]:
        """Recall episodes (delegates to retrieval service)."""
        return await self.retrieval.recall(query, k, context=context)

    async def update_from_outcome(
        self,
        episode_id: UUID,
        outcome: Outcome,
    ) -> None:
        """Update memory (delegates to learning service)."""
        await self.learning.update_from_outcome(episode_id, outcome)

    # ... all other existing methods delegate similarly


def get_episodic_memory(session_id: str | None = None) -> EpisodicMemory:
    """Singleton factory (unchanged for backward compatibility)."""
    # Same as before, returns facade
    ...
```

**Lines**: ~400
**Preserved API**: All 34 existing methods maintained

---

### Refactoring Execution Plan

**Step 1: Create new modules (Week 1, Days 1-3)**
1. Create `episodic_fusion.py` - Extract classes (low risk, no dependencies)
2. Create `episodic_storage.py` - Extract storage methods
3. Create `episodic_retrieval.py` - Extract retrieval methods
4. Create `episodic_learning.py` - Extract learning methods
5. Create `episodic_saga.py` - Extract saga orchestration

**Step 2: Refactor facade (Week 1, Days 4-5)**
1. Modify `episodic.py` to import and delegate
2. Keep all method signatures identical
3. Add deprecation warnings (optional, for future cleanup)

**Step 3: Update tests (Week 2)**
1. Run existing 3,214 lines of episodic tests - should pass unchanged
2. Add unit tests for new services (target: 500 lines)
3. Add integration tests for service coordination (target: 200 lines)

### Testing Requirements

**Backward Compatibility**:
- All 8,075 existing tests must pass unchanged
- No changes to test files in Phase 1

**New Tests** (Phase 1):
```
tests/memory/
├── test_episodic_storage.py         # 150 lines - storage service
├── test_episodic_retrieval.py       # 200 lines - retrieval service
├── test_episodic_learning.py        # 100 lines - learning service
├── test_episodic_fusion.py          # 50 lines - fusion/reranking
└── test_episodic_integration.py     # 200 lines - service coordination
```

**Coverage Target**: 90% for new modules

---

### API Router Refactoring (config.py)

**Problem**: `create_ww_router()` is 429 lines in `/mnt/projects/ww/src/ww/api/routes/config.py`

**Solution**: Extract config model builders

```python
# NEW: src/ww/api/models/config_models.py
"""
Configuration model builders.

Separates Pydantic model construction from API routing logic.
"""

class ConfigModelBuilder:
    """Builder for SystemConfigResponse from settings and runtime config."""

    @staticmethod
    async def build_fsrs_config(settings, runtime: dict) -> FSRSConfig:
        """Build FSRS config section."""
        return FSRSConfig(
            defaultStability=runtime.get("fsrs_default_stability", settings.fsrs_default_stability),
            retentionTarget=runtime.get("fsrs_retention_target", settings.fsrs_retention_target),
            ...
        )

    @staticmethod
    async def build_actr_config(settings, runtime: dict) -> ACTRConfig:
        """Build ACT-R config section."""
        ...

    @staticmethod
    async def build_full_config(settings, runtime: dict) -> SystemConfigResponse:
        """Build complete config response."""
        return SystemConfigResponse(
            fsrs=await ConfigModelBuilder.build_fsrs_config(settings, runtime),
            actr=await ConfigModelBuilder.build_actr_config(settings, runtime),
            ...
        )

# REFACTORED: src/ww/api/routes/config.py (now ~150 lines)
from ww.api.models.config_models import ConfigModelBuilder

@router.get("", response_model=SystemConfigResponse)
async def get_config():
    """Get current system configuration."""
    settings = get_settings()
    async with _runtime_config_lock:
        runtime = _runtime_config.copy()
    return await ConfigModelBuilder.build_full_config(settings, runtime)
```

**Before**: 839 lines
**After**: ~150 lines (routing) + ~300 lines (models) = 450 total, but separated concerns

---

### Phase 1 Deliverables

- [ ] 6 new episodic modules created (2,800 total lines)
- [ ] episodic.py reduced to 400-line facade
- [ ] config.py reduced to 150 lines + new config_models.py
- [ ] All 8,075 existing tests pass
- [ ] 700 new test lines for episodic services
- [ ] Documentation updated (API unchanged, implementation notes added)

**Estimated Effort**: 80 hours (2 weeks, 1 developer)

---

## Phase 2: Caching and Performance

**Duration**: 2 weeks
**Priority**: P0 (Production readiness)
**Dependencies**: Phase 1 (episodic refactoring makes cache integration cleaner)
**Can overlap with**: Hinton work (different modules)

### Problem Analysis

1. **No Redis caching**: Every embedding lookup hits BGE-M3 (expensive)
2. **N+1 graph queries**: `get_relationships()` called in loop for graph expansion
3. **No API rate limiting**: Vulnerable to abuse, DoS

### Redis Caching Strategy

**What to cache**:
1. **Embeddings**: Query embeddings (TTL: 1 hour, size: ~4KB per entry)
2. **Search results**: Top-k results for common queries (TTL: 5 min, size: ~10KB)
3. **Graph relationships**: Node neighbors (TTL: 10 min, size: ~2KB)
4. **Cluster assignments**: Episode → cluster mapping (TTL: 1 hour, size: ~100B)

**What NOT to cache**:
1. **Episode content**: Too large, changes frequently
2. **Learned model weights**: Need persistence, not cache
3. **Session state**: Use Postgres/Neo4j for durability

#### 2.1 Redis Integration

**New Module**: `/mnt/projects/ww/src/ww/storage/redis_cache.py`

```python
"""
Redis caching layer for World Weaver.

Caches embeddings, search results, and graph relationships.
"""

import hashlib
import json
import pickle
from typing import Any

import redis.asyncio as redis
import numpy as np

class RedisCache:
    """
    Async Redis cache with type-specific TTLs.

    Cache Keys:
    - emb:{hash}           - Query embeddings (1h TTL)
    - search:{hash}:{k}    - Search results (5m TTL)
    - graph:{node_id}      - Graph relationships (10m TTL)
    - cluster:{ep_id}      - Cluster assignments (1h TTL)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 50,
    ):
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            decode_responses=False,  # Binary for numpy arrays
        )
        self.client = redis.Redis(connection_pool=self.pool)

        # TTLs (seconds)
        self.TTL_EMBEDDING = 3600  # 1 hour
        self.TTL_SEARCH = 300      # 5 minutes
        self.TTL_GRAPH = 600       # 10 minutes
        self.TTL_CLUSTER = 3600    # 1 hour

    async def get_embedding(self, text: str) -> np.ndarray | None:
        """
        Get cached embedding.

        Complexity: O(1) hash lookup
        """
        key = f"emb:{self._hash_text(text)}"
        data = await self.client.get(key)
        if data is None:
            return None
        return pickle.loads(data)

    async def set_embedding(
        self,
        text: str,
        embedding: np.ndarray
    ) -> None:
        """
        Cache embedding with TTL.

        Storage: ~4KB per 1024-dim float32 embedding
        """
        key = f"emb:{self._hash_text(text)}"
        await self.client.setex(
            key,
            self.TTL_EMBEDDING,
            pickle.dumps(embedding),
        )

    async def get_search_results(
        self,
        query: str,
        k: int
    ) -> list[dict] | None:
        """Get cached search results."""
        key = f"search:{self._hash_text(query)}:{k}"
        data = await self.client.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def set_search_results(
        self,
        query: str,
        k: int,
        results: list[dict],
    ) -> None:
        """Cache search results (short TTL, results may change)."""
        key = f"search:{self._hash_text(query)}:{k}"
        await self.client.setex(
            key,
            self.TTL_SEARCH,
            json.dumps(results, default=str),
        )

    async def get_graph_relationships(
        self,
        node_id: str
    ) -> list[dict] | None:
        """Get cached graph relationships."""
        key = f"graph:{node_id}"
        data = await self.client.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def set_graph_relationships(
        self,
        node_id: str,
        relationships: list[dict],
    ) -> None:
        """Cache graph relationships."""
        key = f"graph:{node_id}"
        await self.client.setex(
            key,
            self.TTL_GRAPH,
            json.dumps(relationships, default=str),
        )

    async def invalidate_search(self, session_id: str) -> None:
        """Invalidate all search caches for session (after new episode)."""
        pattern = f"search:*"
        async for key in self.client.scan_iter(match=pattern):
            await self.client.delete(key)

    async def get_cluster_assignment(
        self,
        episode_id: str
    ) -> int | None:
        """Get cached cluster ID."""
        key = f"cluster:{episode_id}"
        data = await self.client.get(key)
        if data is None:
            return None
        return int(data)

    async def set_cluster_assignment(
        self,
        episode_id: str,
        cluster_id: int,
    ) -> None:
        """Cache cluster assignment."""
        key = f"cluster:{episode_id}"
        await self.client.setex(
            key,
            self.TTL_CLUSTER,
            str(cluster_id),
        )

    def _hash_text(self, text: str) -> str:
        """Hash text for cache key (SHA256, 64 hex chars)."""
        return hashlib.sha256(text.encode()).hexdigest()

    async def close(self) -> None:
        """Close Redis connection."""
        await self.client.close()
        await self.pool.disconnect()


# Singleton factory
_cache: RedisCache | None = None

def get_redis_cache() -> RedisCache:
    """Get Redis cache singleton."""
    global _cache
    if _cache is None:
        settings = get_settings()
        _cache = RedisCache(redis_url=settings.redis_url)
    return _cache
```

**Lines**: ~250
**Dependencies**: `redis[asyncio]` (add to pyproject.toml)

---

#### 2.2 Embedding Provider with Cache

**Modified**: `/mnt/projects/ww/src/ww/embedding/bge_m3.py`

```python
class BGEM3EmbeddingProvider(EmbeddingProvider):
    """BGE-M3 with Redis caching."""

    def __init__(self, ...):
        ...
        self.cache = get_redis_cache() if get_settings().redis_enabled else None

    async def embed(self, text: str) -> np.ndarray:
        """
        Embed with cache-aside pattern.

        Flow:
        1. Check cache: O(1)
        2. If miss, compute: O(n) for text length n
        3. Store in cache: O(1)

        Cache hit rate: ~70-80% for typical workloads (query reuse)
        """
        if self.cache:
            cached = await self.cache.get_embedding(text)
            if cached is not None:
                return cached

        # Cache miss - compute
        embedding = self._model.encode(text, normalize_embeddings=True)

        if self.cache:
            await self.cache.set_embedding(text, embedding)

        return embedding
```

**Impact**: 70-80% cache hit rate → 5x speedup for repeated queries

---

#### 2.3 Graph Store with Batch Queries

**Problem**: `get_relationships()` called in loop during graph expansion

**Current Code** (N+1 pattern):
```python
# In episodic.py:_expand_graph_neighbors()
neighbors = []
for episode_id in episode_ids:  # N iterations
    rels = await self.graph_store.get_relationships(episode_id)  # 1 query each
    neighbors.extend(rels)
```

**Complexity**: O(N) queries, each O(log M) for M total nodes → O(N log M)

**Solution**: Use existing `get_relationships_batch()` method

**Modified**: `/mnt/projects/ww/src/ww/memory/episodic_retrieval.py`

```python
async def _expand_graph_neighbors(
    self,
    episode_ids: list[UUID],
    max_depth: int = 1
) -> list[dict]:
    """
    Graph traversal with batch queries.

    OLD: O(N) queries
    NEW: O(1) batch query + optional cache
    """
    # Check cache first
    if self.cache:
        cached_rels = {}
        uncached_ids = []
        for ep_id in episode_ids:
            rels = await self.cache.get_graph_relationships(str(ep_id))
            if rels:
                cached_rels[str(ep_id)] = rels
            else:
                uncached_ids.append(str(ep_id))
    else:
        uncached_ids = [str(ep_id) for ep_id in episode_ids]
        cached_rels = {}

    # Batch query for uncached
    if uncached_ids:
        batch_rels = await self.graph_store.get_relationships_batch(
            node_ids=uncached_ids,
            rel_type=None,
            direction="both",
        )

        # Cache results
        if self.cache:
            for node_id, rels in batch_rels.items():
                await self.cache.set_graph_relationships(node_id, rels)

        # Merge cached + fresh
        all_rels = {**cached_rels, **batch_rels}
    else:
        all_rels = cached_rels

    # Flatten
    neighbors = []
    for rels in all_rels.values():
        neighbors.extend(rels)

    return neighbors
```

**Complexity**: O(1) batch query (or O(K) cache lookups) → **10-100x speedup**

---

#### 2.4 API Rate Limiting

**New Middleware**: `/mnt/projects/ww/src/ww/api/middleware/rate_limit.py`

```python
"""
Token bucket rate limiting middleware.

Limits:
- Authenticated users: 1000 req/min
- Anonymous users: 100 req/min
- Admin endpoints: 50 req/min (higher cost operations)
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

class TokenBucket:
    """Token bucket algorithm for rate limiting."""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Returns:
            True if allowed, False if rate limited
        """
        now = time.monotonic()
        elapsed = now - self.last_update

        # Refill tokens
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate,
        )
        self.last_update = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with per-user buckets.

    Limits:
    - /api/episodes: 1000/min (16.67/sec)
    - /api/admin: 50/min (0.83/sec)
    - Default: 100/min (1.67/sec)
    """

    def __init__(self, app):
        super().__init__(app)
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(rate=1.67, capacity=100)
        )
        self.bucket_lock = asyncio.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Determine user key (IP or API key)
        user_key = self._get_user_key(request)

        # Determine rate limit
        rate, capacity = self._get_limits(request)

        # Get or create bucket
        async with self.bucket_lock:
            if user_key not in self.buckets:
                self.buckets[user_key] = TokenBucket(rate, capacity)
            bucket = self.buckets[user_key]

        # Check rate limit
        if not bucket.consume():
            return Response(
                content="Rate limit exceeded",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(int(1.0 / rate)),
                    "X-RateLimit-Limit": str(capacity),
                    "X-RateLimit-Remaining": str(int(bucket.tokens)),
                },
            )

        # Proceed
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        return response

    def _get_user_key(self, request: Request) -> str:
        """Get user identifier (API key or IP)."""
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:16]}"  # Truncate for privacy

        # Fall back to IP
        return f"ip:{request.client.host}"

    def _get_limits(self, request: Request) -> tuple[float, int]:
        """
        Get rate and capacity for endpoint.

        Returns:
            (rate per second, capacity)
        """
        path = request.url.path

        # Admin endpoints (low limit)
        if "/admin" in path or "/config" in path:
            return (0.83, 50)  # 50/min

        # Authenticated users (high limit)
        if request.headers.get("X-API-Key"):
            return (16.67, 1000)  # 1000/min

        # Anonymous (default)
        return (1.67, 100)  # 100/min
```

**Lines**: ~150
**Integration**: Add to `/mnt/projects/ww/src/ww/api/server.py`:

```python
from ww.api.middleware.rate_limit import RateLimitMiddleware

app = FastAPI(...)
app.add_middleware(RateLimitMiddleware)
```

**Bypass for tests**: Environment variable `WW_DISABLE_RATE_LIMIT=1`

---

### Phase 2 Deliverables

- [ ] Redis cache layer implemented (250 lines)
- [ ] Embedding provider integrated with cache
- [ ] Graph batch queries with cache
- [ ] Rate limiting middleware (150 lines)
- [ ] Configuration added to settings
- [ ] 300 lines of cache/rate-limit tests
- [ ] Performance benchmarks (expect 5-10x speedup)

**Estimated Effort**: 60 hours (1.5 weeks, 1 developer)

---

## Phase 3: Quality and Observability

**Duration**: 1 week
**Priority**: P1 (Quality improvement)
**Dependencies**: None (can run in parallel with Phase 1-2)
**Can overlap with**: All other work (cleanup task)

### 3.1 Logger Conversion

**Problem**: 232 print() statements in production code

**Strategy**: Automated conversion with safety checks

**Script**: `/mnt/projects/ww/scripts/convert_prints_to_logger.py`

```python
"""
Convert print() statements to logger calls.

Usage:
    python scripts/convert_prints_to_logger.py --check  # Dry run
    python scripts/convert_prints_to_logger.py --apply  # Apply changes
"""

import ast
import re
import sys
from pathlib import Path

class PrintToLoggerConverter(ast.NodeTransformer):
    """AST transformer to convert print() to logger calls."""

    def visit_Call(self, node):
        """Visit function calls and convert print()."""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            # Determine log level from content
            if node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant):
                    content = str(first_arg.value).lower()

                    # Error keywords
                    if any(kw in content for kw in ["error", "failed", "exception"]):
                        level = "error"
                    # Warning keywords
                    elif any(kw in content for kw in ["warning", "warn", "deprecated"]):
                        level = "warning"
                    # Info keywords
                    elif any(kw in content for kw in ["starting", "initialized", "completed"]):
                        level = "info"
                    # Debug default
                    else:
                        level = "debug"
                else:
                    level = "debug"  # Default for dynamic content
            else:
                level = "debug"

            # Create logger.{level}() call
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="logger", ctx=ast.Load()),
                    attr=level,
                    ctx=ast.Load(),
                ),
                args=node.args,
                keywords=node.keywords,
            )

        return self.generic_visit(node)

def convert_file(file_path: Path, dry_run: bool = True) -> dict:
    """Convert print() to logger in file."""
    source = file_path.read_text()
    tree = ast.parse(source)

    # Count prints
    print_count = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    )

    if print_count == 0:
        return {"file": str(file_path), "prints": 0, "converted": 0}

    # Check if logger import exists
    has_logger = "import logging" in source or "from logging import" in source

    # Transform
    converter = PrintToLoggerConverter()
    new_tree = converter.visit(tree)

    if not dry_run:
        # Add logger if needed
        if not has_logger:
            # Find first import
            for i, line in enumerate(source.split("\n")):
                if line.startswith("import ") or line.startswith("from "):
                    # Insert after imports
                    lines = source.split("\n")
                    lines.insert(i + 1, "import logging")
                    lines.insert(i + 2, "")
                    lines.insert(i + 3, "logger = logging.getLogger(__name__)")
                    source = "\n".join(lines)
                    break

        # Write back (using ast.unparse for Python 3.9+)
        # Note: This is simplified - real implementation needs black formatting
        # file_path.write_text(ast.unparse(new_tree))

    return {
        "file": str(file_path),
        "prints": print_count,
        "converted": print_count if not dry_run else 0,
        "had_logger": has_logger,
    }

def main():
    src_dir = Path(__file__).parent.parent / "src" / "ww"
    results = []

    for py_file in src_dir.rglob("*.py"):
        result = convert_file(py_file, dry_run="--check" in sys.argv)
        if result["prints"] > 0:
            results.append(result)

    # Summary
    total_prints = sum(r["prints"] for r in results)
    total_converted = sum(r["converted"] for r in results)

    print(f"Found {total_prints} print() statements in {len(results)} files")
    if "--apply" in sys.argv:
        print(f"Converted {total_converted} to logger calls")
    else:
        print("Run with --apply to convert")

if __name__ == "__main__":
    main()
```

**Alternative (Manual)**: Use regex + manual review

```bash
# Find all print() statements
grep -rn "print(" src/ww --include="*.py" | grep -v "# noqa" > prints.txt

# Manual conversion pattern:
# Before: print(f"Starting {service}")
# After:  logger.info(f"Starting {service}")
```

**Priority Files** (highest impact):
1. `/mnt/projects/ww/src/ww/nca/*.py` (17 files with prints)
2. `/mnt/projects/ww/src/ww/interfaces/*.py` (9 files - user-facing)
3. `/mnt/projects/ww/src/ww/bridges/*.py` (3 files)

**Effort**: ~20 hours (manual review + testing)

---

### 3.2 Bridge Test Coverage

**Problem**: Only 5 bridge test files, need comprehensive coverage

**Target**: 20+ test files covering all bridge integration points

**New Tests**:

```
tests/bridges/
├── test_capsule_bridge.py                    # ✓ Exists (13KB)
├── test_dopamine_bridge.py                   # ✓ Exists (11KB)
├── test_ff_encoding_bridge.py                # ✓ Exists (12KB)
├── test_ff_retrieval_scorer.py               # ✓ Exists (9KB)
├── test_nca_bridge.py                        # ✓ Exists (16KB)
├── test_glymphatic_bridge.py                 # NEW - Glymphatic integration
├── test_consolidation_bridge.py              # NEW - Consolidation coupling
├── test_hippocampus_bridge.py                # NEW - HPC-episodic bridge
├── test_vta_bridge.py                        # NEW - VTA-dopamine bridge
├── test_adenosine_bridge.py                  # NEW - Sleep-wake regulation
├── test_pattern_separation_bridge.py         # NEW - DG integration
├── test_neuromod_orchestra_bridge.py         # NEW - Multi-modulator coordination
├── test_learned_gate_bridge.py               # NEW - Gate-storage integration
├── test_buffer_manager_bridge.py             # NEW - Buffer-episodic flow
├── test_three_factor_bridge.py               # NEW - Learning rule integration
└── test_end_to_end_learning.py               # NEW - Full pipeline test
```

**Coverage Target**: 85% line coverage for all bridge modules

**Estimated Lines**: 2,500 new test lines

---

### 3.3 Documentation Updates

**Files to Update**:

1. `/mnt/projects/ww/docs/architecture.md` - Add refactoring section
2. `/mnt/projects/ww/docs/API_WALKTHROUGH.md` - Update cache examples
3. `/mnt/projects/ww/README.md` - Add Redis setup
4. `/mnt/projects/ww/docs/ROADMAP.md` - Mark Phase 1-3 complete
5. NEW: `/mnt/projects/ww/docs/CACHING_GUIDE.md` - Redis best practices
6. NEW: `/mnt/projects/ww/docs/RATE_LIMITING.md` - API limits guide

---

### Phase 3 Deliverables

- [ ] 232 print() statements converted to logger
- [ ] 15 new bridge test files (2,500 lines)
- [ ] Bridge coverage: 85%+
- [ ] Documentation updated (6 files)
- [ ] Code quality score: 9.0/10 (vs 7.7 current)

**Estimated Effort**: 40 hours (1 week, 1 developer)

---

## Testing Strategy

### Regression Testing (All Phases)

**Critical**: All 8,075 existing tests must pass after each phase

**Test Execution**:
```bash
# Full test suite (before and after each phase)
pytest tests/ -v --cov=src/ww --cov-report=html

# Target: 90%+ coverage maintained
# Existing: 80% coverage baseline
```

**Automated Checks**:
```yaml
# .github/workflows/refactoring-ci.yml
name: Refactoring CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src/ww

      - name: Check coverage
        run: |
          coverage report --fail-under=80

      - name: Check test count
        run: |
          # Ensure test count doesn't decrease
          TEST_COUNT=$(pytest --collect-only | grep "tests collected" | awk '{print $1}')
          if [ "$TEST_COUNT" -lt 8075 ]; then
            echo "Test count decreased!"
            exit 1
          fi
```

---

### Performance Benchmarks

**New Benchmarks**: `/mnt/projects/ww/tests/benchmarks/test_refactoring_performance.py`

```python
"""
Performance benchmarks for refactoring validation.

Ensures refactoring improves or maintains performance.
"""

import pytest
from uuid import uuid4

@pytest.mark.benchmark(group="episodic-store")
def test_store_performance(benchmark, episodic_memory):
    """Benchmark episode storage (should be ~same after refactoring)."""

    def store_episode():
        return episodic_memory.store(
            content="Test episode",
            context=EpisodeContext(...),
        )

    result = benchmark(store_episode)

    # Target: < 50ms (unchanged from before)
    assert benchmark.stats["mean"] < 0.05


@pytest.mark.benchmark(group="episodic-recall")
def test_recall_with_cache(benchmark, episodic_memory):
    """Benchmark recall with Redis cache (should be 5x faster)."""

    # Pre-populate cache
    episodic_memory.recall("test query", k=10)

    def recall_cached():
        return episodic_memory.recall("test query", k=10)

    result = benchmark(recall_cached)

    # Target: < 10ms with cache (vs ~50ms without)
    assert benchmark.stats["mean"] < 0.01


@pytest.mark.benchmark(group="graph-traversal")
def test_batch_graph_query(benchmark, neo4j_store):
    """Benchmark batch graph queries (should be 10-100x faster)."""

    # Create test nodes
    node_ids = [str(uuid4()) for _ in range(100)]

    def batch_query():
        return neo4j_store.get_relationships_batch(node_ids)

    result = benchmark(batch_query)

    # Target: < 100ms for 100 nodes (vs ~2s for N+1)
    assert benchmark.stats["mean"] < 0.1
```

**Run Benchmarks**:
```bash
# Before refactoring (baseline)
pytest tests/benchmarks -v --benchmark-save=before

# After Phase 2 (expect improvements)
pytest tests/benchmarks -v --benchmark-save=after

# Compare
pytest-benchmark compare before after
```

---

## Migration Guide

### For Users (API Unchanged)

**No migration required** - All public APIs maintain backward compatibility.

**Optional Updates**:
1. Add Redis to deployment: `docker-compose.yml` update (provided below)
2. Set `REDIS_URL` environment variable
3. Enable rate limiting (automatic with middleware)

---

### For Developers (Internal Changes)

**Phase 1 (Episodic Refactoring)**:

```python
# OLD: Direct access to EpisodicMemory internals
from ww.memory.episodic import EpisodicMemory
em = EpisodicMemory()
em.vector_store.add(...)  # Direct access

# NEW: Use public API (preferred) or access services
from ww.memory.episodic import EpisodicMemory
em = EpisodicMemory()
em.storage.store(...)  # Service-oriented

# Facade maintains backward compat
em.vector_store.add(...)  # Still works (deprecated but not removed)
```

**Phase 2 (Caching)**:

```python
# Configuration (add to .env or settings)
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379

# Automatic cache integration (no code changes)
# Embedding provider uses cache transparently
```

**Phase 3 (Logging)**:

```python
# OLD: print() statements
print(f"Processing {item}")

# NEW: Logger calls
logger.debug(f"Processing {item}")

# Tests that capture stdout will need update
# Use caplog fixture instead of capsys
```

---

### Docker Compose Update

**File**: `/mnt/projects/ww/docker-compose.yml`

```yaml
services:
  # Existing services (neo4j, qdrant, api)
  ...

  # NEW: Redis cache
  redis:
    image: redis:7-alpine
    container_name: ww-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped
    networks:
      - ww-network

volumes:
  redis-data:
    driver: local
  # Existing volumes...

networks:
  ww-network:
    driver: bridge
```

**Environment Variables** (`.env`):
```bash
# Redis configuration
REDIS_ENABLED=true
REDIS_URL=redis://redis:6379
REDIS_MAX_CONNECTIONS=50

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_AUTHENTICATED=1000  # per minute
RATE_LIMIT_ANONYMOUS=100       # per minute
```

---

## Success Metrics

### Phase 1: Episodic Decomposition

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| episodic.py lines | 3,616 | ≤ 400 | `wc -l episodic.py` |
| Largest class | 3,139 | ≤ 500 | AST analysis |
| Module count | 1 | 6 | File count |
| Test pass rate | 100% | 100% | pytest |
| Coverage | 80% | 90% | pytest-cov |

### Phase 2: Caching & Performance

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| Query latency (cached) | 50ms | ≤ 10ms | Benchmark |
| Embedding reuse rate | 0% | 70-80% | Cache hit metrics |
| Graph traversal (100 nodes) | ~2s | ≤ 100ms | Benchmark |
| API rate limit | None | 100-1000/min | Load test |
| Redis memory usage | N/A | ≤ 2GB | `redis-cli INFO` |

### Phase 3: Quality & Observability

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| print() statements | 232 | 0 | `grep -r "print("` |
| Bridge test files | 5 | 20+ | File count |
| Bridge coverage | ~60% | 85%+ | pytest-cov |
| Code quality score | 7.7/10 | 9.0/10 | Architecture review |
| Documentation files | 42 | 48 | Doc count |

---

## Risk Assessment

### High Risk Items

1. **Episodic refactoring breaks tests**
   - **Mitigation**: Facade pattern preserves API, incremental migration
   - **Rollback**: Git revert, restore episodic.py from tag

2. **Redis introduces caching bugs**
   - **Mitigation**: Cache invalidation on writes, short TTLs, feature flag
   - **Rollback**: Set `REDIS_ENABLED=false`

3. **Rate limiting blocks legitimate users**
   - **Mitigation**: High limits (1000/min), monitoring, bypass for authenticated
   - **Rollback**: Disable middleware

### Medium Risk Items

1. **Performance regressions from service indirection**
   - **Mitigation**: Benchmarks in CI, profiling
   - **Acceptable**: ≤ 5% overhead for cleaner architecture

2. **Logger conversion changes output**
   - **Mitigation**: Manual review of critical logs, test updates
   - **Note**: Some tests may need caplog instead of capsys

### Low Risk Items

1. **Documentation drift**
   - **Mitigation**: Doc updates in same PR as code

2. **Bridge test coverage gaps**
   - **Mitigation**: Property-based testing, mutation testing

---

## Dependencies and Scheduling

### Dependency Graph

```
Phase 1 (Episodic)
    ↓
Phase 2 (Caching) ← Cleaner to integrate after refactoring
    ↓
Phase 3 (Quality) ← Can run in parallel with 1 & 2
```

### Parallelization Strategy

**Week 1-2**: Phase 1 (Episodic)
- **Can overlap with**: CompBio work (different modules)
- **Blocks**: Phase 2 caching integration

**Week 3-4**: Phase 2 (Caching)
- **Can overlap with**: Hinton work (different modules)
- **Requires**: Phase 1 complete (cleaner integration points)

**Week 1-5**: Phase 3 (Quality) - Continuous
- **Logger conversion**: Week 1-2 (during Phase 1)
- **Bridge tests**: Week 3-4 (during Phase 2)
- **Docs**: Week 5 (final polish)
- **Can overlap with**: All phases (cleanup work)

### Critical Path

**Total Duration**: 5 weeks (with overlap)
- **Sequential duration**: 5 weeks
- **Parallel duration**: ~4 weeks (with 2 developers)

**Blockers**:
- Phase 2 depends on Phase 1 (service interfaces)
- Final release depends on all 3 phases

---

## Effort Estimation

### Developer Hours

| Phase | Task | Hours | Developer |
|-------|------|-------|-----------|
| **Phase 1** | Episodic decomposition | 60 | Backend |
| | API router refactoring | 12 | Backend |
| | Testing & validation | 8 | QA |
| **Phase 2** | Redis integration | 24 | Backend |
| | Cache middleware | 12 | Backend |
| | Batch query optimization | 8 | Backend |
| | Rate limiting | 8 | Backend |
| | Performance testing | 8 | QA |
| **Phase 3** | Logger conversion | 20 | Any |
| | Bridge tests | 16 | QA |
| | Documentation | 4 | Tech Writer |
| **Total** | | **180 hours** | ~4.5 weeks @ 40h/week |

### Resource Allocation

**1 Developer (Sequential)**: 5 weeks
**2 Developers (Parallel)**: 3 weeks
- Developer 1: Phase 1 + Phase 2
- Developer 2: Phase 3 (parallel)

---

## Rollback Plan

### Phase 1 Rollback

**Trigger**: Tests fail, performance regression > 10%

**Steps**:
1. `git revert <phase1-commits>`
2. Restore `episodic.py` from `v0.5.0` tag
3. Re-run tests to confirm stability
4. Investigate failures in separate branch

**Recovery Time**: < 1 hour

---

### Phase 2 Rollback

**Trigger**: Cache corruption, memory issues, rate limit problems

**Steps**:
1. Disable Redis: `REDIS_ENABLED=false`
2. Disable rate limiting: Comment out middleware in `server.py`
3. Restart API server
4. Monitor for stability
5. Fix issues and re-enable incrementally

**Recovery Time**: < 30 minutes (no code changes needed)

---

### Phase 3 Rollback

**Trigger**: Logger issues, test failures

**Steps**:
1. Revert logger changes (if needed): `git revert <phase3-commits>`
2. Bridge tests are additive (no rollback needed)
3. Documentation can be updated later

**Recovery Time**: < 1 hour

---

## Post-Refactoring Validation

### Checklist

- [ ] All 8,075+ tests pass
- [ ] Coverage ≥ 80% (no regression)
- [ ] Performance benchmarks meet targets
- [ ] API documentation updated
- [ ] Docker Compose includes Redis
- [ ] Migration guide published
- [ ] Rollback procedures tested
- [ ] Code quality score ≥ 9.0/10
- [ ] Zero print() statements in src/
- [ ] Bridge coverage ≥ 85%

### Production Readiness

**Post-Phase 1**:
- ✅ Cleaner architecture
- ✅ Better maintainability
- ⚠️ Still missing caching (performance)

**Post-Phase 2**:
- ✅ Production-ready performance
- ✅ Rate limiting enabled
- ✅ Redis caching operational

**Post-Phase 3**:
- ✅ Code quality best practices
- ✅ Comprehensive test coverage
- ✅ Documentation complete

**Target**: v0.6.0 release after all 3 phases

---

## Conclusion

This refactoring plan addresses all 7 critical architectural issues while maintaining 100% backward compatibility with existing tests. The phased approach allows parallel work with CompBio and Hinton initiatives.

**Key Benefits**:
1. **Maintainability**: God object eliminated, clear separation of concerns
2. **Performance**: 5-10x speedup from caching and batch queries
3. **Quality**: Logger standardization, 85%+ bridge coverage
4. **Production Readiness**: Rate limiting, monitoring, rollback procedures

**Next Steps**:
1. Review and approve plan
2. Create Phase 1 branch
3. Begin episodic decomposition
4. Run continuous integration checks

---

## Appendix: File Paths Reference

### New Files Created

**Phase 1**:
- `/mnt/projects/ww/src/ww/memory/episodic_storage.py`
- `/mnt/projects/ww/src/ww/memory/episodic_retrieval.py`
- `/mnt/projects/ww/src/ww/memory/episodic_learning.py`
- `/mnt/projects/ww/src/ww/memory/episodic_fusion.py`
- `/mnt/projects/ww/src/ww/memory/episodic_saga.py`
- `/mnt/projects/ww/src/ww/api/models/config_models.py`

**Phase 2**:
- `/mnt/projects/ww/src/ww/storage/redis_cache.py`
- `/mnt/projects/ww/src/ww/api/middleware/rate_limit.py`

**Phase 3**:
- `/mnt/projects/ww/scripts/convert_prints_to_logger.py`
- `/mnt/projects/ww/tests/bridges/test_glymphatic_bridge.py`
- `/mnt/projects/ww/tests/bridges/test_consolidation_bridge.py`
- (+ 13 more bridge test files)

### Modified Files

**Phase 1**:
- `/mnt/projects/ww/src/ww/memory/episodic.py` (3,616 → 400 lines)
- `/mnt/projects/ww/src/ww/api/routes/config.py` (839 → 150 lines)

**Phase 2**:
- `/mnt/projects/ww/src/ww/embedding/bge_m3.py` (add cache)
- `/mnt/projects/ww/src/ww/api/server.py` (add middleware)
- `/mnt/projects/ww/docker-compose.yml` (add Redis)

**Phase 3**:
- 50+ files with print() statements converted to logger

### Test Files

**Phase 1**:
- `/mnt/projects/ww/tests/memory/test_episodic_storage.py` (NEW)
- `/mnt/projects/ww/tests/memory/test_episodic_retrieval.py` (NEW)
- `/mnt/projects/ww/tests/memory/test_episodic_learning.py` (NEW)
- `/mnt/projects/ww/tests/memory/test_episodic_integration.py` (NEW)

**Phase 2**:
- `/mnt/projects/ww/tests/storage/test_redis_cache.py` (NEW)
- `/mnt/projects/ww/tests/api/test_rate_limit.py` (NEW)
- `/mnt/projects/ww/tests/benchmarks/test_refactoring_performance.py` (NEW)

**Phase 3**:
- 15 new bridge test files

---

**Document Version**: 1.0
**Last Updated**: 2026-01-07
**Status**: Ready for Review
