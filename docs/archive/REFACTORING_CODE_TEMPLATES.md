# Architecture Refactoring Code Templates

**Reference**: [ARCHITECTURE_REFACTORING_PLAN.md](/mnt/projects/t4d/t4dm/docs/ARCHITECTURE_REFACTORING_PLAN.md)

This document provides copy-paste code templates for critical refactoring tasks.

---

## Phase 1: Episodic Storage Service Template

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic_storage.py`

```python
"""
Episodic Memory Storage Layer.

Handles create/update/delete operations with saga pattern for dual-store consistency.

Complexity:
- store(): O(1) storage + O(log n) index update
- store_batch(): O(k) for k episodes
- update_embedding(): O(1) update
"""

import logging
from datetime import datetime
from uuid import UUID, uuid4

import numpy as np

from t4dm.core.types import Episode, EpisodeContext
from t4dm.storage.t4dx_graph_adapter import T4DXGraphAdapter
from t4dm.storage.t4dx_vector_adapter import T4DXVectorAdapter
from t4dm.storage.saga import Saga, SagaState

logger = logging.getLogger(__name__)


class EpisodicStorageService:
    """Storage service for episodic memories with saga-based consistency."""

    def __init__(
        self,
        vector_store: T4DXVectorAdapter,
        graph_store: T4DXGraphAdapter,
        session_id: str,
    ):
        """
        Initialize storage service.

        Args:
            vector_store: Qdrant vector database
            graph_store: Neo4j graph database
            session_id: Session namespace for isolation
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.session_id = session_id

        # Temporal sequencing
        self._last_episode_id: UUID | None = None
        self._sequence_counter: int = 0

    async def store(
        self,
        episode: Episode,
        embedding: np.ndarray,
    ) -> UUID:
        """
        Store episode with saga pattern.

        Args:
            episode: Episode to store
            embedding: Dense embedding vector

        Returns:
            Episode ID

        Raises:
            RuntimeError: If saga fails
        """
        # Create saga for dual-store consistency
        saga = Saga(saga_id=str(uuid4()))

        # Prepare payloads
        episode_payload = {
            "id": str(episode.id),
            "content": episode.content,
            "timestamp": episode.timestamp.isoformat(),
            "session_id": self.session_id,
            "context": episode.context.model_dump() if episode.context else {},
            "outcome": episode.outcome.model_dump() if episode.outcome else None,
        }

        # Step 1: Add to vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.vector_store.add(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
                vectors=[embedding],
                payloads=[episode_payload],
            ),
            compensate=lambda: self.vector_store.delete(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
            ),
        )

        # Step 2: Create graph node
        graph_props = self._to_graph_props(episode)
        saga.add_step(
            name="create_node",
            action=lambda: self.graph_store.create_node(
                label="Episode",
                properties=graph_props,
            ),
            compensate=lambda: self.graph_store.delete_node(
                node_id=str(episode.id),
                label="Episode",
            ),
        )

        # Execute saga
        result = await saga.execute()

        # Check result
        if result.state != SagaState.COMMITTED:
            raise RuntimeError(
                f"Episode storage failed: {result.error} "
                f"(saga: {result.saga_id}, state: {result.state.value})"
            )

        logger.info(
            f"Stored episode {episode.id} in session {self.session_id} "
            f"(saga: {result.saga_id})"
        )

        # Update temporal sequencing
        if self._last_episode_id is not None:
            try:
                await self._link_episodes(self._last_episode_id, episode.id)
            except Exception as e:
                logger.warning(f"Failed to link episodes: {e}")

        self._last_episode_id = episode.id
        self._sequence_counter += 1

        return episode.id

    async def store_batch(
        self,
        episodes: list[Episode],
        embeddings: list[np.ndarray],
    ) -> list[UUID]:
        """
        Batch storage for consolidation.

        Args:
            episodes: Episodes to store
            embeddings: Corresponding embeddings

        Returns:
            List of episode IDs
        """
        if len(episodes) != len(embeddings):
            raise ValueError("Episodes and embeddings length mismatch")

        # Batch vectors
        ids = [str(ep.id) for ep in episodes]
        payloads = [
            {
                "id": str(ep.id),
                "content": ep.content,
                "timestamp": ep.timestamp.isoformat(),
                "session_id": self.session_id,
            }
            for ep in episodes
        ]

        await self.vector_store.add(
            collection=self.vector_store.episodes_collection,
            ids=ids,
            vectors=embeddings,
            payloads=payloads,
        )

        # Batch graph nodes (sequential for now, can be parallelized)
        for episode in episodes:
            props = self._to_graph_props(episode)
            await self.graph_store.create_node(label="Episode", properties=props)

        logger.info(f"Batch stored {len(episodes)} episodes")

        return [ep.id for ep in episodes]

    async def update_embedding(
        self,
        episode_id: UUID,
        new_embedding: np.ndarray,
    ) -> None:
        """
        Update episode embedding (reconsolidation).

        Args:
            episode_id: Episode to update
            new_embedding: New embedding vector
        """
        await self.vector_store.update(
            collection=self.vector_store.episodes_collection,
            ids=[str(episode_id)],
            vectors=[new_embedding],
        )
        logger.debug(f"Updated embedding for episode {episode_id}")

    async def delete(self, episode_id: UUID) -> None:
        """
        Delete episode with saga rollback.

        Args:
            episode_id: Episode to delete
        """
        saga = Saga(saga_id=str(uuid4()))

        saga.add_step(
            name="delete_vector",
            action=lambda: self.vector_store.delete(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode_id)],
            ),
            compensate=lambda: None,  # Cannot restore deleted vector
        )

        saga.add_step(
            name="delete_node",
            action=lambda: self.graph_store.delete_node(
                node_id=str(episode_id),
                label="Episode",
            ),
            compensate=lambda: None,
        )

        result = await saga.execute()

        if result.state != SagaState.COMMITTED:
            logger.error(f"Episode deletion failed: {result.error}")
            raise RuntimeError(f"Deletion failed: {result.error}")

        logger.info(f"Deleted episode {episode_id}")

    async def _link_episodes(
        self,
        prev_id: UUID,
        next_id: UUID,
    ) -> None:
        """
        Create temporal link between episodes.

        Args:
            prev_id: Previous episode
            next_id: Next episode
        """
        await self.graph_store.create_relationship(
            from_node_id=str(prev_id),
            to_node_id=str(next_id),
            rel_type="TEMPORAL_BEFORE",
            properties={"sequence": self._sequence_counter},
        )

    def _to_graph_props(self, episode: Episode) -> dict:
        """Convert episode to graph properties."""
        props = {
            "id": str(episode.id),
            "content": episode.content[:500],  # Truncate for graph
            "timestamp": episode.timestamp.isoformat(),
            "session_id": self.session_id,
            "created_at": datetime.utcnow().isoformat(),
        }

        if episode.context:
            props["context_type"] = episode.context.type
            props["context_project"] = episode.context.project

        if episode.outcome:
            props["outcome_success"] = episode.outcome.success

        return props
```

---

## Phase 2: Redis Cache Template

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/storage/redis_cache.py`

```python
"""
Redis caching layer for T4DM.

Caches embeddings, search results, and graph relationships with TTL-based expiration.

Cache Keys:
- emb:{hash}           - Query embeddings (1h TTL)
- search:{hash}:{k}    - Search results (5m TTL)
- graph:{node_id}      - Graph relationships (10m TTL)
- cluster:{ep_id}      - Cluster assignments (1h TTL)
"""

import hashlib
import json
import logging
import pickle
from typing import Any

import numpy as np
import redis.asyncio as redis

from t4dm.core.config import get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Async Redis cache with type-specific TTLs.

    Thread-safe connection pooling for concurrent access.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 50,
    ):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            max_connections: Maximum connection pool size
        """
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

        logger.info(f"Redis cache initialized: {redis_url}")

    async def get_embedding(self, text: str) -> np.ndarray | None:
        """
        Get cached embedding.

        Args:
            text: Query text

        Returns:
            Cached embedding or None if miss
        """
        key = f"emb:{self._hash_text(text)}"
        try:
            data = await self.client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def set_embedding(
        self,
        text: str,
        embedding: np.ndarray,
    ) -> None:
        """
        Cache embedding with TTL.

        Args:
            text: Query text
            embedding: Embedding vector
        """
        key = f"emb:{self._hash_text(text)}"
        try:
            await self.client.setex(
                key,
                self.TTL_EMBEDDING,
                pickle.dumps(embedding),
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def get_search_results(
        self,
        query: str,
        k: int,
    ) -> list[dict] | None:
        """
        Get cached search results.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Cached results or None if miss
        """
        key = f"search:{self._hash_text(query)}:{k}"
        try:
            data = await self.client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def set_search_results(
        self,
        query: str,
        k: int,
        results: list[dict],
    ) -> None:
        """
        Cache search results (short TTL, results may change).

        Args:
            query: Search query
            k: Number of results
            results: Search results
        """
        key = f"search:{self._hash_text(query)}:{k}"
        try:
            await self.client.setex(
                key,
                self.TTL_SEARCH,
                json.dumps(results, default=str),
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def get_graph_relationships(
        self,
        node_id: str,
    ) -> list[dict] | None:
        """
        Get cached graph relationships.

        Args:
            node_id: Node ID

        Returns:
            Cached relationships or None if miss
        """
        key = f"graph:{node_id}"
        try:
            data = await self.client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def set_graph_relationships(
        self,
        node_id: str,
        relationships: list[dict],
    ) -> None:
        """
        Cache graph relationships.

        Args:
            node_id: Node ID
            relationships: Relationship list
        """
        key = f"graph:{node_id}"
        try:
            await self.client.setex(
                key,
                self.TTL_GRAPH,
                json.dumps(relationships, default=str),
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def invalidate_search(self, pattern: str = "search:*") -> int:
        """
        Invalidate search caches (e.g., after new episode stored).

        Args:
            pattern: Key pattern to match

        Returns:
            Number of keys deleted
        """
        count = 0
        try:
            async for key in self.client.scan_iter(match=pattern):
                await self.client.delete(key)
                count += 1
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
        return count

    def _hash_text(self, text: str) -> str:
        """
        Hash text for cache key.

        Args:
            text: Text to hash

        Returns:
            SHA256 hex digest (64 chars)
        """
        return hashlib.sha256(text.encode()).hexdigest()

    async def close(self) -> None:
        """Close Redis connection."""
        await self.client.close()
        await self.pool.disconnect()
        logger.info("Redis cache closed")


# Singleton factory
_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    """
    Get Redis cache singleton.

    Returns:
        RedisCache instance
    """
    global _cache
    if _cache is None:
        settings = get_settings()
        if not settings.redis_enabled:
            raise RuntimeError("Redis not enabled in settings")
        _cache = RedisCache(
            redis_url=settings.redis_url,
            max_connections=settings.redis_max_connections,
        )
    return _cache
```

---

## Phase 2: Rate Limiting Middleware Template

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/api/middleware/rate_limit.py`

```python
"""
Token bucket rate limiting middleware.

Limits:
- Authenticated users: 1000 req/min
- Anonymous users: 100 req/min
- Admin endpoints: 50 req/min
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Refills tokens at constant rate up to capacity.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

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

    Uses token bucket algorithm with different limits per endpoint type.
    """

    def __init__(self, app):
        super().__init__(app)
        self.buckets: dict[str, TokenBucket] = {}
        self.bucket_lock = asyncio.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware

        Returns:
            Response (or 429 if rate limited)
        """
        # Determine user key
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
            logger.warning(f"Rate limit exceeded for {user_key}: {request.url.path}")
            return Response(
                content="Rate limit exceeded",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(int(1.0 / rate)),
                    "X-RateLimit-Limit": str(capacity),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Proceed with request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(capacity)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))

        return response

    def _get_user_key(self, request: Request) -> str:
        """
        Get user identifier for rate limiting.

        Args:
            request: Request object

        Returns:
            User key (API key or IP address)
        """
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:16]}"  # Truncate for privacy

        # Fall back to IP
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _get_limits(self, request: Request) -> tuple[float, int]:
        """
        Get rate and capacity for endpoint.

        Args:
            request: Request object

        Returns:
            Tuple of (rate per second, capacity)
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

---

## Phase 2: Embedding Provider with Cache Integration

**File**: Modify `/mnt/projects/t4d/t4dm/src/t4dm/embedding/bge_m3.py`

```python
# Add at top of file:
from t4dm.storage.redis_cache import get_redis_cache

# Modify __init__:
class BGEM3EmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_path: str = "BAAI/bge-m3", device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device

        # Load model
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(model_path, use_fp16=True)

        # Cache integration (optional)
        settings = get_settings()
        self.cache = get_redis_cache() if settings.redis_enabled else None

        logger.info(f"BGE-M3 loaded on {device}, cache: {self.cache is not None}")

    async def embed(self, text: str) -> np.ndarray:
        """
        Embed with cache-aside pattern.

        Flow:
        1. Check cache: O(1)
        2. If miss, compute: O(n) for text length n
        3. Store in cache: O(1)
        """
        # Check cache
        if self.cache:
            cached = await self.cache.get_embedding(text)
            if cached is not None:
                return cached

        # Cache miss - compute
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )["dense_vecs"]

        # Store in cache
        if self.cache:
            await self.cache.set_embedding(text, embedding)

        return embedding
```

---

## Phase 3: Print to Logger Conversion Examples

### Before

```python
# Bad: print() statement
print(f"Processing episode {episode_id}")
print(f"ERROR: Failed to store: {error}")
print(f"Warning: Cache miss for {key}")
```

### After

```python
# Good: Logger with appropriate level
logger.debug(f"Processing episode {episode_id}")
logger.error(f"Failed to store: {error}")
logger.warning(f"Cache miss for {key}")
```

### Template

```python
import logging

logger = logging.getLogger(__name__)

# Debug: Detailed flow information
logger.debug(f"Variable state: {state}")

# Info: High-level operations
logger.info(f"Started service {service_name}")

# Warning: Recoverable issues
logger.warning(f"Cache miss, falling back to DB")

# Error: Failures that need attention
logger.error(f"Operation failed: {error}")

# Critical: System-level failures
logger.critical(f"Database connection lost")
```

---

## Testing Template: Bridge Tests

**File**: `/mnt/projects/t4d/t4dm/tests/bridges/test_glymphatic_bridge.py`

```python
"""
Tests for Glymphatic-Consolidation bridge integration.

Validates sleep-wake cycle coupling with memory consolidation.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from t4dm.consolidation.sleep import SleepCycleManager
from t4dm.nca.glymphatic import GlymphaticSystem
from t4dm.memory.episodic import EpisodicMemory


@pytest.fixture
def glymphatic_system():
    """Create glymphatic system."""
    return GlymphaticSystem(
        adenosine_threshold=0.7,
        clearance_rate=0.5,
    )


@pytest.fixture
def sleep_manager():
    """Create sleep cycle manager."""
    return SleepCycleManager(
        cycle_duration_hours=8.0,
    )


@pytest.fixture
def episodic_memory(session_id):
    """Create episodic memory service."""
    return EpisodicMemory(session_id=session_id)


@pytest.mark.asyncio
async def test_sleep_triggers_consolidation(
    glymphatic_system,
    sleep_manager,
    episodic_memory,
):
    """Test that sleep cycle triggers memory consolidation."""
    # Store test episodes
    for i in range(10):
        await episodic_memory.store(
            content=f"Test episode {i}",
            context=None,
        )

    # Simulate wake period (adenosine accumulation)
    for _ in range(16):  # 16 hours awake
        glymphatic_system.update(dt=1.0)  # 1 hour steps

    # Check adenosine level
    assert glymphatic_system.adenosine_level > 0.7

    # Trigger sleep
    sleep_manager.enter_sleep()

    # Verify glymphatic activation
    clearance = glymphatic_system.get_clearance_rate()
    assert clearance > 0.3

    # Verify consolidation triggered
    # (Would check consolidation service state here)


@pytest.mark.asyncio
async def test_clearance_rate_modulates_consolidation(
    glymphatic_system,
):
    """Test that glymphatic clearance rate affects consolidation speed."""
    # Low clearance (awake)
    glymphatic_system.set_state("awake")
    clearance_awake = glymphatic_system.get_clearance_rate()

    # High clearance (sleep)
    glymphatic_system.set_state("sleep")
    clearance_sleep = glymphatic_system.get_clearance_rate()

    # Sleep should have 2-3x higher clearance
    assert clearance_sleep > clearance_awake * 2


@pytest.mark.asyncio
async def test_adenosine_gates_consolidation(
    glymphatic_system,
    sleep_manager,
):
    """Test that high adenosine is required for consolidation."""
    # Low adenosine (recently woke)
    glymphatic_system.adenosine_level = 0.2

    # Should not trigger consolidation
    assert not sleep_manager.should_consolidate()

    # High adenosine (sleep deprived)
    glymphatic_system.adenosine_level = 0.9

    # Should trigger consolidation
    assert sleep_manager.should_consolidate()


# Add 10+ more test cases covering:
# - Clearance rate decay during wake
# - Waste accumulation limits
# - Multi-cycle consolidation
# - Integration with SWR replay
# - Spindle-ripple coordination
```

---

## Configuration Template: Settings Updates

**File**: Modify `/mnt/projects/t4d/t4dm/src/t4dm/core/config.py`

```python
class Settings(BaseSettings):
    # Existing settings...

    # Phase 2: Redis caching
    redis_enabled: bool = Field(default=False, description="Enable Redis caching")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_max_connections: int = Field(default=50, description="Max Redis connections")

    # Phase 2: Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_authenticated: int = Field(default=1000, description="Rate limit for authenticated users (per minute)")
    rate_limit_anonymous: int = Field(default=100, description="Rate limit for anonymous users (per minute)")
    rate_limit_admin: int = Field(default=50, description="Rate limit for admin endpoints (per minute)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="T4DM_",
    )
```

---

## Docker Compose Template: Redis Service

**File**: Modify `/mnt/projects/t4d/t4dm/docker-compose.yml`

```yaml
version: "3.8"

services:
  # Existing services (neo4j, qdrant, api)...

  redis:
    image: redis:7-alpine
    container_name: ww-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: |
      redis-server
      --appendonly yes
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --save 60 1000
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped
    networks:
      - ww-network

volumes:
  # Existing volumes...
  redis-data:
    driver: local

networks:
  ww-network:
    driver: bridge
```

---

## Testing Template: Performance Benchmarks

**File**: `/mnt/projects/t4d/t4dm/tests/benchmarks/test_refactoring_performance.py`

```python
"""
Performance benchmarks for refactoring validation.

Ensures refactoring improves or maintains performance.
"""

import pytest
import numpy as np
from uuid import uuid4

from t4dm.memory.episodic import EpisodicMemory
from t4dm.core.types import Episode, EpisodeContext


@pytest.fixture
def episodic_memory(session_id):
    """Create episodic memory service."""
    return EpisodicMemory(session_id=session_id)


@pytest.mark.benchmark(group="episodic-store")
def test_store_performance(benchmark, episodic_memory):
    """Benchmark episode storage (should be ~same after refactoring)."""

    async def store_episode():
        return await episodic_memory.store(
            content="Test episode content for benchmarking",
            context=EpisodeContext(type="test", project="benchmark"),
        )

    result = benchmark.pedantic(
        lambda: asyncio.run(store_episode()),
        rounds=10,
        iterations=5,
    )

    # Target: < 50ms (unchanged from before)
    assert benchmark.stats["mean"] < 0.05


@pytest.mark.benchmark(group="episodic-recall")
def test_recall_with_cache(benchmark, episodic_memory):
    """Benchmark recall with Redis cache (should be 5x faster)."""

    # Pre-populate cache
    asyncio.run(episodic_memory.recall("test query", k=10))

    async def recall_cached():
        return await episodic_memory.recall("test query", k=10)

    result = benchmark.pedantic(
        lambda: asyncio.run(recall_cached()),
        rounds=10,
        iterations=10,
    )

    # Target: < 10ms with cache (vs ~50ms without)
    assert benchmark.stats["mean"] < 0.01


@pytest.mark.benchmark(group="graph-traversal")
def test_batch_graph_query(benchmark, t4dx_graph_adapter):
    """Benchmark batch graph queries (should be 10-100x faster)."""

    # Create test nodes
    node_ids = [str(uuid4()) for _ in range(100)]

    async def batch_query():
        return await t4dx_graph_adapter.get_relationships_batch(node_ids)

    result = benchmark.pedantic(
        lambda: asyncio.run(batch_query()),
        rounds=5,
        iterations=3,
    )

    # Target: < 100ms for 100 nodes (vs ~2s for N+1)
    assert benchmark.stats["mean"] < 0.1
```

---

## Migration Checklist

### Phase 1: Episodic Refactoring

- [ ] Create `episodic_storage.py` (copy template)
- [ ] Create `episodic_retrieval.py` (copy template)
- [ ] Create `episodic_learning.py`
- [ ] Create `episodic_fusion.py` (extract from episodic.py)
- [ ] Create `episodic_saga.py`
- [ ] Refactor `episodic.py` to facade
- [ ] Run tests: `pytest tests/memory/test_episodic*.py -v`
- [ ] Create new service tests
- [ ] Update documentation

### Phase 2: Caching & Performance

- [ ] Create `redis_cache.py` (copy template)
- [ ] Create `rate_limit.py` (copy template)
- [ ] Modify `bge_m3.py` (add cache integration)
- [ ] Modify `server.py` (add middleware)
- [ ] Update `docker-compose.yml` (add Redis)
- [ ] Update `config.py` (add settings)
- [ ] Run benchmarks: `pytest tests/benchmarks -v`
- [ ] Test rate limiting
- [ ] Verify cache hit rates

### Phase 3: Quality & Observability

- [ ] Convert print() to logger (run script or manual)
- [ ] Create 15 new bridge test files
- [ ] Run coverage: `pytest --cov=src/t4dm/bridges`
- [ ] Update documentation
- [ ] Final validation: all 8,075+ tests pass

---

**Document Version**: 1.0
**Last Updated**: 2026-01-07
**Status**: Ready for Implementation
