"""
Qdrant Vector Store for World Weaver.

Provides vector storage and similarity search for memory embeddings.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from opentelemetry.trace import SpanKind
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from t4dm.core.config import get_settings
from t4dm.observability.tracing import traced
from t4dm.storage.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)

# Default timeout for database operations (seconds)
DEFAULT_DB_TIMEOUT = 30

# Phase 2C: Capsule pose payload field names
# Per Hinton (2017, 2018): Pose matrices encode configuration, not just existence
CAPSULE_PAYLOAD_FIELDS = [
    "capsule_poses",           # Flattened 4x4 pose matrices per capsule
    "capsule_activations",     # Capsule activation strengths (existence probability)
    "capsule_routing_agreement",  # Routing agreement score (consistency of votes)
    "capsule_mean_activation",  # Mean activation for quick filtering
]


@dataclass
class CapsulePoseData:
    """
    Phase 2C: Capsule pose data retrieved from storage.

    Encapsulates pose matrices, activations, and routing agreement
    for use in retrieval confidence scoring.

    Per Hinton critique: These fields enable content-addressable memory
    based on pose similarity, not just vector similarity.
    """
    poses: list[float] | None = None  # Flattened pose matrices
    activations: list[float] | None = None  # Per-capsule activations
    routing_agreement: float | None = None  # Agreement score [0, 1]
    mean_activation: float | None = None  # Mean activation

    @property
    def has_poses(self) -> bool:
        """Check if pose data is available."""
        return self.poses is not None and len(self.poses) > 0

    @property
    def has_activations(self) -> bool:
        """Check if activation data is available."""
        return self.activations is not None and len(self.activations) > 0


class DatabaseTimeoutError(Exception):
    """Raised when a database operation times out."""

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Database operation '{operation}' timed out after {timeout}s")


class QdrantStore:
    """
    Qdrant vector store implementation.

    Manages collections for episodic, semantic, and procedural memory.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        hybrid_prefetch_multiplier: float = 1.5,
        dimension: int | None = None,
    ):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            api_key: Optional API key
            timeout: Operation timeout in seconds (default: 30)
            circuit_breaker_config: Optional circuit breaker configuration
            hybrid_prefetch_multiplier: Prefetch factor for hybrid search (default 1.5).
                Higher values (2.0) improve recall but increase network transfer.
                Lower values (1.2) reduce transfer but may miss relevant results.
            dimension: Expected embedding dimension (default: from settings)
        """
        settings = get_settings()

        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.dimension = dimension if dimension is not None else settings.embedding_dimension
        self.timeout = timeout or DEFAULT_DB_TIMEOUT
        # P2-OPT-B2.2: Configurable prefetch multiplier for hybrid search
        self.hybrid_prefetch_multiplier = max(1.0, hybrid_prefetch_multiplier)

        # Collection names
        self.episodes_collection = settings.qdrant_collection_episodes
        self.entities_collection = settings.qdrant_collection_entities
        self.procedures_collection = settings.qdrant_collection_procedures

        self._client: AsyncQdrantClient | None = None
        self._sync_client: QdrantClient | None = None

        # QDRANT-004 FIX: Locks for thread-safe lazy initialization
        self._client_lock: asyncio.Lock | None = None
        self._client_lock_guard = threading.Lock()  # Guards creation of _client_lock

        # Circuit breaker for resilience
        cb_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            reset_timeout=60.0,
        )
        self._circuit_breaker = get_circuit_breaker("qdrant", cb_config)

        # ATOM-P3-23: Track embedding norm statistics for adversarial detection
        self._norm_stats = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}

    async def _with_timeout(self, coro, operation: str):
        """Execute coroutine with timeout and circuit breaker protection."""
        async def _execute():
            try:
                async with asyncio.timeout(self.timeout):
                    return await coro
            except TimeoutError:
                logger.error(f"Timeout in Qdrant operation '{operation}' after {self.timeout}s")
                raise DatabaseTimeoutError(operation, self.timeout)

        return await self._circuit_breaker.execute(_execute)

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this store."""
        return self._circuit_breaker

    def _get_init_lock(self) -> asyncio.Lock:
        """
        Get or create async initialization lock (thread-safe).

        QDRANT-004 FIX: Uses double-checked locking pattern to ensure
        the asyncio.Lock is created exactly once, even with concurrent access.
        """
        if self._client_lock is None:
            with self._client_lock_guard:
                # Double-check after acquiring thread lock
                if self._client_lock is None:
                    self._client_lock = asyncio.Lock()
        return self._client_lock

    async def _get_client(self) -> AsyncQdrantClient:
        """
        Get or create async client (thread-safe).

        QDRANT-004 FIX: Uses asyncio.Lock to prevent race condition
        where multiple coroutines could create multiple clients.
        """
        # Fast path: client already initialized
        if self._client is not None:
            return self._client

        # Slow path: need to initialize with lock
        async with self._get_init_lock():
            # Double-check after acquiring lock
            if self._client is None:
                self._client = AsyncQdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                )
            return self._client

    def _get_sync_client(self) -> QdrantClient:
        """
        Get or create sync client for setup operations.

        Note: Sync client uses threading.Lock for synchronization since
        it's called from sync contexts. Uses the same lock guard.
        """
        if self._sync_client is None:
            with self._client_lock_guard:
                # Double-check after acquiring lock
                if self._sync_client is None:
                    self._sync_client = QdrantClient(
                        url=self.url,
                        api_key=self.api_key,
                    )
        return self._sync_client

    async def initialize(self) -> None:
        """Initialize collections if they don't exist."""
        client = await self._get_client()

        collections = [
            self.episodes_collection,
            self.entities_collection,
            self.procedures_collection,
        ]

        for collection_name in collections:
            await self._ensure_collection(client, collection_name)

    async def _ensure_collection(
        self,
        client: AsyncQdrantClient,
        name: str,
        hybrid: bool = False,
    ) -> None:
        """
        Create collection if it doesn't exist.

        Args:
            client: Qdrant client
            name: Collection name
            hybrid: If True, create with named vectors (dense + sparse)
        """
        try:
            await client.get_collection(name)
            logger.debug(f"Collection '{name}' exists")
        except UnexpectedResponse:
            logger.info(f"Creating collection '{name}' (hybrid={hybrid})")

            if hybrid:
                # Hybrid collection with named dense and sparse vectors
                await client.create_collection(
                    collection_name=name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.dimension,
                            distance=models.Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False),
                        ),
                    },
                )
            else:
                # Standard dense-only collection
                await client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )

        # P2-OPT-B2.1: Ensure session_id payload index exists for efficient filtering
        await self._ensure_session_id_index(client, name)

    async def _ensure_session_id_index(
        self,
        client: AsyncQdrantClient,
        name: str,
    ) -> None:
        """
        Ensure session_id payload index exists for efficient filtering.

        P2-OPT-B2.1 FIX: Without a payload index, session_id filtering
        requires O(n) full collection scan. With index, it's O(log n).

        Args:
            client: Qdrant client
            name: Collection name
        """
        try:
            # Check if index already exists
            collection_info = await client.get_collection(name)
            payload_schema = collection_info.payload_schema or {}

            if "session_id" in payload_schema:
                logger.debug(f"session_id index exists for '{name}'")
                return

            # Create the index
            logger.info(f"Creating session_id index for '{name}'")
            await client.create_payload_index(
                collection_name=name,
                field_name="session_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created session_id index for '{name}'")

        except Exception as e:
            # Non-fatal: index creation failure shouldn't break operations
            logger.warning(f"Failed to create session_id index for '{name}': {e}")

    async def _add_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Add a single batch of vectors to collection.

        Args:
            collection: Collection name
            ids: Vector IDs (UUIDs as strings)
            vectors: Embedding vectors
            payloads: Metadata for each vector
        """
        client = await self._get_client()

        # ATOM-P2-24: Validate embedding dimensions
        for i, vector in enumerate(vectors):
            vec_array = np.asarray(vector) if not isinstance(vector, np.ndarray) else vector
            if not np.all(np.isfinite(vec_array)):
                raise ValueError(f"Embedding at index {i} contains NaN or Inf")
            if len(vec_array) != self.dimension:
                raise ValueError(
                    f"Embedding at index {i} has dimension {len(vec_array)}, "
                    f"expected {self.dimension}"
                )

            # ATOM-P3-23: Adversarial embedding detection via norm validation
            norm = float(np.linalg.norm(vec_array))
            self._norm_stats['count'] += 1
            self._norm_stats['sum'] += norm
            self._norm_stats['sum_sq'] += norm ** 2

            if self._norm_stats['count'] > 100:
                mean_norm = self._norm_stats['sum'] / self._norm_stats['count']
                if norm > mean_norm * 10 or norm < mean_norm * 0.01:
                    raise ValueError(
                        f"Adversarial embedding detected at index {i}: "
                        f"norm={norm:.4f}, mean={mean_norm:.4f}"
                    )

        points = [
            models.PointStruct(
                id=id_str,
                vector=vector,
                payload=payload,
            )
            for id_str, vector, payload in zip(ids, vectors, payloads)
        ]

        await client.upsert(
            collection_name=collection,
            points=points,
        )

        logger.debug(f"Added {len(points)} vectors to '{collection}'")

    async def add(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> None:
        """
        Add vectors with payloads to collection.

        For large batches (> batch_size), splits into chunks and uploads in parallel
        for improved performance. Provides rollback on partial failure.

        Args:
            collection: Collection name
            ids: Vector IDs (UUIDs as strings)
            vectors: Embedding vectors
            payloads: Metadata for each vector
            batch_size: Maximum size per batch (default: 100)
            max_concurrency: Maximum parallel uploads (default: 10)

        Complexity:
            Time: O(n) where n = len(ids)
            Space: O(batch_size) for chunk storage
        """
        async def _add():
            if len(ids) <= batch_size:
                # Single batch - direct upload
                await self._add_batch(collection, ids, vectors, payloads)
            else:
                # Large batch - split into chunks and upload in parallel
                chunks = []
                for i in range(0, len(ids), batch_size):
                    chunk_ids = ids[i:i + batch_size]
                    chunk_vecs = vectors[i:i + batch_size]
                    chunk_payloads = payloads[i:i + batch_size]
                    chunks.append((chunk_ids, chunk_vecs, chunk_payloads))

                logger.debug(f"Splitting {len(ids)} vectors into {len(chunks)} parallel batches")

                # QDRANT-001 FIX: Limit parallelism to prevent database overload
                semaphore = asyncio.Semaphore(max_concurrency)

                # QDRANT-002 FIX: Track completed IDs for precise rollback
                completed_ids: list[str] = []
                completed_lock = asyncio.Lock()

                async def add_with_limit(chunk_ids, chunk_vecs, chunk_payloads):
                    """Upload batch with semaphore limiting, tracking success for rollback."""
                    async with semaphore:
                        await self._add_batch(collection, chunk_ids, chunk_vecs, chunk_payloads)
                        # Track successful chunk IDs (thread-safe)
                        async with completed_lock:
                            completed_ids.extend(chunk_ids)

                try:
                    # Upload chunks with limited concurrency
                    tasks = [
                        add_with_limit(chunk_ids, chunk_vecs, chunk_payloads)
                        for chunk_ids, chunk_vecs, chunk_payloads in chunks
                    ]

                    # Execute with gather to get all results or fail on first error
                    await asyncio.gather(*tasks)

                    logger.info(f"Successfully added {len(ids)} vectors in {len(chunks)} parallel batches")

                except Exception as e:
                    # QDRANT-002 FIX: Only rollback successfully uploaded vectors
                    logger.error(f"Parallel batch add failed: {e}. Attempting rollback...")

                    if completed_ids:
                        client = await self._get_client()
                        try:
                            await client.delete(
                                collection_name=collection,
                                points_selector=models.PointIdsList(points=completed_ids),
                            )
                            logger.warning(f"Rolled back {len(completed_ids)} of {len(ids)} vectors from '{collection}'")
                        except Exception as rollback_err:
                            logger.error(f"Rollback failed for '{collection}': {rollback_err}")
                            logger.error("Database may be in inconsistent state - manual cleanup needed")
                    else:
                        logger.info("No vectors to rollback (none were successfully uploaded)")

                    raise

        return await self._with_timeout(_add(), f"add({collection})")

    @traced("qdrant.search", kind=SpanKind.CLIENT)
    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        session_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Search for similar vectors.

        Uses Qdrant's native filtering to apply constraints BEFORE vector similarity
        computation for maximum efficiency. session_id filtering is particularly important
        for multi-session memory isolation.

        Args:
            collection: Collection name
            vector: Query vector
            limit: Maximum results
            filter: Qdrant filter conditions
            session_id: Optional session ID for prefiltering (recommended for efficiency)
            score_threshold: Minimum similarity score

        Returns:
            List of (id, score, payload) tuples

        Complexity:
            Time: O(log n + k) where n = total vectors, k = results with filtering
                  O(n) without filtering (full scan)
            Space: O(k) for results
        """
        async def _search():
            client = await self._get_client()

            # Build filter, merging session_id if provided
            query_filter = None
            if filter or session_id:
                # Start with provided filter or empty dict
                filter_dict = filter.copy() if filter else {}

                # Add session_id prefilter for efficiency
                # STORAGE-HIGH-002 FIX: Warn if session_id would override filter value
                if session_id:
                    if "session_id" in filter_dict and filter_dict["session_id"] != session_id:
                        logger.warning(
                            f"session_id parameter '{session_id}' overrides filter "
                            f"session_id '{filter_dict['session_id']}'"
                        )
                    filter_dict["session_id"] = session_id
                    logger.debug(f"Applying session_id prefilter: {session_id}")

                query_filter = self._build_filter(filter_dict)

            results = await client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            return [
                (str(hit.id), hit.score, hit.payload or {})
                for hit in results.points
            ]

        return await self._with_timeout(_search(), f"search({collection})")

    @traced("qdrant.search_hybrid", kind=SpanKind.CLIENT)
    async def search_hybrid(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        session_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Hybrid search using Qdrant's native prefetch + RRF fusion.

        Combines dense (semantic) and sparse (lexical) search for better
        recall on both conceptual and exact-match queries.

        Args:
            collection: Collection name
            dense_vector: Dense embedding vector (1024-dim)
            sparse_vector: Sparse vector as {token_id: weight}
            limit: Maximum results
            filter: Qdrant filter conditions
            session_id: Optional session ID for prefiltering
            score_threshold: Minimum similarity score

        Returns:
            List of (id, score, payload) tuples ranked by RRF fusion
        """
        async def _search_hybrid():
            client = await self._get_client()

            # Build filter, merging session_id if provided
            query_filter = None
            if filter or session_id:
                filter_dict = filter.copy() if filter else {}
                # STORAGE-HIGH-002 FIX: Warn if session_id would override filter value
                if session_id:
                    if "session_id" in filter_dict and filter_dict["session_id"] != session_id:
                        logger.warning(
                            f"session_id parameter '{session_id}' overrides filter "
                            f"session_id '{filter_dict['session_id']}'"
                        )
                    filter_dict["session_id"] = session_id
                    logger.debug(f"Hybrid search with session_id prefilter: {session_id}")
                query_filter = self._build_filter(filter_dict)

            # P2-OPT-B2.2: Use configurable prefetch multiplier
            # Default 1.5x balances recall vs network transfer (was 2x)
            prefetch_limit = max(limit, int(limit * self.hybrid_prefetch_multiplier))

            # Build prefetch for both vector types
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ]

            # Only add sparse prefetch if we have sparse weights
            if sparse_vector:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                )

            # Query with RRF fusion
            results = await client.query_points(
                collection_name=collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            return [
                (str(hit.id), hit.score, hit.payload or {})
                for hit in results.points
            ]

        return await self._with_timeout(_search_hybrid(), f"search_hybrid({collection})")

    async def add_hybrid(
        self,
        collection: str,
        ids: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
        payloads: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """
        Add vectors with both dense and sparse representations.

        Args:
            collection: Collection name
            ids: Vector IDs (UUIDs as strings)
            dense_vectors: Dense embedding vectors
            sparse_vectors: Sparse vectors as {token_id: weight} dicts
            payloads: Metadata for each vector
            batch_size: Maximum size per batch
        """
        async def _add_hybrid():
            client = await self._get_client()

            points = []
            for i, (id_str, dense_vec, sparse_vec, payload) in enumerate(
                zip(ids, dense_vectors, sparse_vectors, payloads)
            ):
                point = models.PointStruct(
                    id=id_str,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=list(sparse_vec.keys()) if sparse_vec else [],
                            values=list(sparse_vec.values()) if sparse_vec else [],
                        ),
                    },
                    payload=payload,
                )
                points.append(point)

            # Batch upsert
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await client.upsert(
                    collection_name=collection,
                    points=batch,
                )
                logger.debug(f"Added {len(batch)} hybrid vectors to '{collection}'")

        return await self._with_timeout(_add_hybrid(), f"add_hybrid({collection})")

    async def ensure_hybrid_collection(
        self,
        name: str,
    ) -> None:
        """
        Ensure a hybrid collection exists with named vectors.

        This is a migration helper to upgrade existing collections
        or create new ones with hybrid support.

        Args:
            name: Collection name
        """
        client = await self._get_client()
        await self._ensure_collection(client, name, hybrid=True)

    def _build_filter(self, filter_dict: dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from dict."""
        conditions = []

        for key, value in filter_dict.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            elif isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                            ),
                        )
                    )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=conditions)

    async def get(
        self,
        collection: str,
        ids: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Get vectors by ID.

        Args:
            collection: Collection name
            ids: Vector IDs

        Returns:
            List of (id, payload) tuples
        """
        async def _get():
            client = await self._get_client()

            results = await client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=True,
            )

            return [
                (str(point.id), point.payload or {})
                for point in results
            ]

        return await self._with_timeout(_get(), f"get({collection})")

    async def get_with_vectors(
        self,
        collection: str,
        ids: list[str],
    ) -> list[tuple[str, dict[str, Any], list[float] | None]]:
        """
        Get vectors by ID including the vector data.

        Used for reconsolidation where we need to update embeddings.

        Args:
            collection: Collection name
            ids: Vector IDs

        Returns:
            List of (id, payload, vector) tuples
        """
        async def _get():
            client = await self._get_client()

            results = await client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=True,
                with_vectors=True,
            )

            return [
                (str(point.id), point.payload or {}, point.vector)
                for point in results
            ]

        return await self._with_timeout(_get(), f"get_with_vectors({collection})")

    async def get_capsule_poses(
        self,
        collection: str,
        episode_id: str,
    ) -> CapsulePoseData | None:
        """
        Phase 2C: Retrieve capsule pose data for an episode.

        Per Hinton (2017, 2018): Pose matrices encode entity configuration,
        enabling part-whole composition in memory retrieval. Routing agreement
        measures consistency of capsule votes - higher agreement indicates
        more coherent compositional structure.

        Args:
            collection: Collection name (typically episodes)
            episode_id: Episode UUID as string

        Returns:
            CapsulePoseData with poses, activations, and routing agreement,
            or None if episode not found or has no capsule data.

        Example:
            >>> pose_data = await store.get_capsule_poses("episodes", episode_id)
            >>> if pose_data and pose_data.has_poses:
            ...     # Use routing_agreement for confidence scoring
            ...     confidence_boost = pose_data.routing_agreement * 0.1
        """
        async def _get_poses():
            client = await self._get_client()

            results = await client.retrieve(
                collection_name=collection,
                ids=[episode_id],
                with_payload=True,
            )

            if not results:
                logger.debug(f"Episode {episode_id} not found in {collection}")
                return None

            payload = results[0].payload or {}

            # Extract capsule fields from payload
            poses = payload.get("capsule_poses")
            activations = payload.get("capsule_activations")
            routing_agreement = payload.get("capsule_routing_agreement")
            mean_activation = payload.get("capsule_mean_activation")

            # Return None if no capsule data at all
            if poses is None and activations is None and routing_agreement is None:
                logger.debug(f"No capsule data for episode {episode_id}")
                return None

            return CapsulePoseData(
                poses=poses,
                activations=activations,
                routing_agreement=routing_agreement,
                mean_activation=mean_activation,
            )

        return await self._with_timeout(_get_poses(), f"get_capsule_poses({collection})")

    async def batch_get_capsule_poses(
        self,
        collection: str,
        episode_ids: list[str],
    ) -> dict[str, CapsulePoseData | None]:
        """
        Phase 2C: Batch retrieve capsule pose data for multiple episodes.

        Efficient batch retrieval for confidence scoring during retrieval.

        Args:
            collection: Collection name
            episode_ids: List of episode UUIDs as strings

        Returns:
            Dict mapping episode_id to CapsulePoseData (or None if no data)
        """
        if not episode_ids:
            return {}

        async def _batch_get():
            client = await self._get_client()

            results = await client.retrieve(
                collection_name=collection,
                ids=episode_ids,
                with_payload=True,
            )

            # Map results by ID
            result_dict: dict[str, CapsulePoseData | None] = {}
            for point in results:
                point_id = str(point.id)
                payload = point.payload or {}

                poses = payload.get("capsule_poses")
                activations = payload.get("capsule_activations")
                routing_agreement = payload.get("capsule_routing_agreement")
                mean_activation = payload.get("capsule_mean_activation")

                if poses is None and activations is None and routing_agreement is None:
                    result_dict[point_id] = None
                else:
                    result_dict[point_id] = CapsulePoseData(
                        poses=poses,
                        activations=activations,
                        routing_agreement=routing_agreement,
                        mean_activation=mean_activation,
                    )

            # Fill in missing IDs with None
            for eid in episode_ids:
                if eid not in result_dict:
                    result_dict[eid] = None

            return result_dict

        return await self._with_timeout(_batch_get(), f"batch_get_capsule_poses({collection})")

    async def batch_update_vectors(
        self,
        collection: str,
        updates: list[tuple[str, list[float]]],
    ) -> int:
        """
        Batch update vectors while preserving payloads.

        Used for reconsolidation to update embeddings based on retrieval outcomes.

        Args:
            collection: Collection name
            updates: List of (id, new_vector) tuples

        Returns:
            Number of vectors updated
        """
        if not updates:
            return 0

        async def _batch_update():
            client = await self._get_client()

            # Qdrant requires re-upserting to update vectors
            # First get existing payloads
            ids = [u[0] for u in updates]
            existing = await client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=True,
            )

            # Map id -> payload
            id_to_payload = {str(p.id): p.payload or {} for p in existing}

            # Build points with new vectors but existing payloads
            points = []
            for id_str, new_vector in updates:
                if id_str in id_to_payload:
                    points.append(
                        models.PointStruct(
                            id=id_str,
                            vector=new_vector,
                            payload=id_to_payload[id_str],
                        )
                    )

            if not points:
                return 0

            # Upsert to update vectors
            await client.upsert(
                collection_name=collection,
                points=points,
            )

            logger.debug(f"Updated {len(points)} vectors in '{collection}'")
            return len(points)

        return await self._with_timeout(_batch_update(), f"batch_update_vectors({collection})")

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """
        Delete vectors by ID.

        Args:
            collection: Collection name
            ids: Vector IDs to delete
        """
        async def _delete():
            client = await self._get_client()

            await client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=ids),
            )

            logger.debug(f"Deleted {len(ids)} vectors from '{collection}'")

        return await self._with_timeout(_delete(), f"delete({collection})")

    async def update_payload(
        self,
        collection: str,
        id: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Update payload for a vector.

        Args:
            collection: Collection name
            id: Vector ID
            payload: New/updated payload fields
        """
        async def _update_payload():
            client = await self._get_client()

            await client.set_payload(
                collection_name=collection,
                payload=payload,
                points=[id],
            )

        return await self._with_timeout(_update_payload(), f"update_payload({collection})")

    async def count(
        self,
        collection: str,
        count_filter: dict[str, Any] | None = None,
    ) -> int:
        """
        Get count of vectors in collection with optional filter.

        Args:
            collection: Collection name
            count_filter: Optional filter conditions

        Returns:
            Count of matching vectors
        """
        async def _count():
            client = await self._get_client()

            if count_filter:
                # Use count_points with filter for filtered counts
                query_filter = self._build_filter(count_filter)
                result = await client.count(
                    collection_name=collection,
                    count_filter=query_filter,
                )
                return result.count
            # Use collection info for total count (faster)
            info = await client.get_collection(collection)
            return info.points_count or 0

        return await self._with_timeout(_count(), f"count({collection})")

    async def scroll(
        self,
        collection: str,
        scroll_filter: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[tuple[str, dict[str, Any], list[float] | None]], int | None]:
        """
        Scroll through collection with filter and pagination.

        Args:
            collection: Collection name
            scroll_filter: Optional filter conditions
            limit: Maximum results per page
            offset: Pagination offset (deprecated in favor of scroll API)
            with_payload: Include payloads in results
            with_vectors: Include vectors in results

        Returns:
            Tuple of (results, next_offset) where results are (id, payload, vector) tuples
        """
        async def _scroll():
            client = await self._get_client()

            # Build filter if provided
            query_filter = None
            if scroll_filter:
                query_filter = self._build_filter(scroll_filter)

            # Use scroll API with offset (Qdrant scroll returns offset-based pagination)
            results, next_offset = await client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            formatted = []
            for point in results:
                vector = None
                if with_vectors and point.vector:
                    vector = point.vector if isinstance(point.vector, list) else list(point.vector)
                formatted.append((
                    str(point.id),
                    point.payload or {} if with_payload else {},
                    vector,
                ))

            return formatted, next_offset

        return await self._with_timeout(_scroll(), f"scroll({collection})")

    # Batch Operations

    async def batch_add(
        self,
        operations: list[tuple[str, list[str], list[list[float]], list[dict[str, Any]]]],
    ) -> dict[str, int]:
        """
        Add vectors to multiple collections in a batch.

        Note: Qdrant doesn't support cross-collection transactions,
        so this provides best-effort atomicity with compensation on failure.

        Args:
            operations: List of (collection, ids, vectors, payloads) tuples

        Returns:
            Dict of collection -> count of vectors added

        Raises:
            Exception: If any operation fails (partial completion may occur)
        """
        async def _batch_add():
            client = await self._get_client()
            results = {}
            completed = []

            try:
                for collection, ids, vectors, payloads in operations:
                    points = [
                        models.PointStruct(
                            id=id_str,
                            vector=vector,
                            payload=payload,
                        )
                        for id_str, vector, payload in zip(ids, vectors, payloads)
                    ]

                    await client.upsert(
                        collection_name=collection,
                        points=points,
                    )

                    completed.append((collection, ids))
                    results[collection] = len(points)
                    logger.debug(f"Batch added {len(points)} vectors to '{collection}'")

                return results

            except Exception as e:
                # Compensation: attempt to rollback completed operations
                logger.error(f"Batch add failed: {e}. Attempting compensation...")
                for collection, ids in completed:
                    try:
                        await client.delete(
                            collection_name=collection,
                            points_selector=models.PointIdsList(points=ids),
                        )
                        logger.debug(f"Compensated: deleted {len(ids)} from '{collection}'")
                    except Exception as comp_err:
                        logger.error(f"Compensation failed for '{collection}': {comp_err}")
                raise

        return await self._with_timeout(_batch_add(), "batch_add")

    async def batch_update_payloads(
        self,
        collection: str,
        updates: list[tuple[str, dict[str, Any]]],
        max_concurrency: int = 10,
    ) -> int:
        """
        Update multiple payloads with parallel execution.

        Uses a semaphore to limit concurrent updates and prevent
        overwhelming the database.

        Args:
            collection: Collection name
            updates: List of (id, payload) tuples to update
            max_concurrency: Maximum parallel updates (default 10)

        Returns:
            Count of successful updates

        Note:
            Failed updates are logged but don't stop other updates.
            Check logs for partial failure details.
        """
        if not updates:
            return 0

        async def _batch_update():
            client = await self._get_client()
            semaphore = asyncio.Semaphore(max_concurrency)

            async def update_one(id_str: str, payload: dict) -> bool:
                """Update single payload with semaphore limiting."""
                async with semaphore:
                    try:
                        await client.set_payload(
                            collection_name=collection,
                            payload=payload,
                            points=[id_str],
                            wait=True,
                        )
                        return True
                    except Exception as e:
                        logger.warning(
                            f"Failed to update payload for {id_str} in {collection}: {e}"
                        )
                        return False

            # Execute all updates in parallel with limited concurrency
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*[
                update_one(id_str, payload)
                for id_str, payload in updates
            ])
            elapsed = asyncio.get_event_loop().time() - start_time

            success_count = sum(1 for r in results if r)
            failure_count = len(results) - success_count

            logger.debug(
                f"Batch updated {success_count}/{len(updates)} payloads "
                f"in '{collection}' ({elapsed:.2f}s, {failure_count} failures)"
            )

            if failure_count > 0:
                logger.warning(
                    f"Batch update had {failure_count} failures in '{collection}'"
                )

            return success_count

        return await self._with_timeout(
            _batch_update(),
            f"batch_update_payloads({collection}, {len(updates)} updates)"
        )

    async def batch_delete(
        self,
        collection: str,
        ids: list[str],
        max_concurrency: int = 10,
    ) -> int:
        """
        Delete multiple points with parallel execution.

        Args:
            collection: Collection name
            ids: Point IDs to delete
            max_concurrency: Maximum parallel deletes

        Returns:
            Count of successful deletes
        """
        if not ids:
            return 0

        async def _batch_delete():
            client = await self._get_client()
            semaphore = asyncio.Semaphore(max_concurrency)

            async def delete_one(id_str: str) -> bool:
                async with semaphore:
                    try:
                        await client.delete(
                            collection_name=collection,
                            points_selector=models.PointIdsList(
                                points=[id_str],
                            ),
                            wait=True,
                        )
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to delete {id_str}: {e}")
                        return False

            results = await asyncio.gather(*[
                delete_one(id_str) for id_str in ids
            ])

            return sum(1 for r in results if r)

        return await self._with_timeout(
            _batch_delete(),
            f"batch_delete({collection}, {len(ids)} ids)"
        )

    async def batch_delete_multi(
        self,
        deletions: list[tuple[str, list[str]]],
    ) -> dict[str, int]:
        """
        Delete vectors from multiple collections.

        Args:
            deletions: List of (collection, ids) tuples

        Returns:
            Dict of collection -> count of vectors deleted
        """
        async def _batch_delete():
            client = await self._get_client()
            results = {}

            for collection, ids in deletions:
                await client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=ids),
                )
                results[collection] = len(ids)
                logger.debug(f"Batch deleted {len(ids)} vectors from '{collection}'")

            return results

        return await self._with_timeout(_batch_delete(), "batch_delete_multi")

    async def close(self) -> None:
        """Close client connections."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None


# Thread-safe singleton management
import asyncio

_qdrant_instances: dict[str, QdrantStore] = {}
_qdrant_lock = threading.Lock()
# BUG-004 FIX: Use loop-specific locks to handle event loop changes correctly
# STORAGE-HIGH-003 FIX: Store (loop, lock) tuples to enable cleanup of closed loops
_qdrant_async_locks: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}
_qdrant_locks_lock = threading.Lock()  # Thread-safe access to locks dict


def _cleanup_closed_loops() -> None:
    """
    STORAGE-HIGH-003 FIX: Remove locks for closed event loops.

    Must be called while holding _qdrant_locks_lock.
    """
    closed_ids = [
        loop_id for loop_id, (loop, _) in _qdrant_async_locks.items()
        if loop.is_closed()
    ]
    for loop_id in closed_ids:
        del _qdrant_async_locks[loop_id]
    if closed_ids:
        logger.debug(f"Cleaned up {len(closed_ids)} stale async locks")


def _get_async_lock() -> asyncio.Lock:
    """
    Get or create async lock for the current event loop (must be called from async context).

    BUG-004 FIX: Use a dictionary keyed by event loop ID instead of a global lock
    to properly handle multiple event loops (e.g., in pytest-asyncio testing).

    STORAGE-HIGH-003 FIX: Clean up locks for closed event loops to prevent memory leak.
    """
    try:
        current_loop = asyncio.get_running_loop()
        loop_id = id(current_loop)
    except RuntimeError:
        # No running event loop - shouldn't happen in async context
        raise RuntimeError("_get_async_lock must be called from async context")

    # Thread-safe lock creation for the specific event loop
    with _qdrant_locks_lock:
        # STORAGE-HIGH-003 FIX: Clean up closed loops before adding new entry
        _cleanup_closed_loops()

        if loop_id not in _qdrant_async_locks:
            _qdrant_async_locks[loop_id] = (current_loop, asyncio.Lock())
        return _qdrant_async_locks[loop_id][1]


def get_qdrant_store(session_id: str = "default") -> QdrantStore:
    """
    Get or create singleton Qdrant store (thread-safe).

    Args:
        session_id: Session identifier for instance isolation

    Returns:
        QdrantStore instance for the session
    """
    if session_id not in _qdrant_instances:
        with _qdrant_lock:
            # Double-check locking pattern
            if session_id not in _qdrant_instances:
                _qdrant_instances[session_id] = QdrantStore()
                logger.debug(f"Created QdrantStore for session: {session_id}")
    return _qdrant_instances[session_id]


async def close_qdrant_store(session_id: str | None = None) -> None:
    """
    Close Qdrant store connection(s).

    STORAGE-CRITICAL-003 FIX: Uses threading lock (not just async lock) to
    synchronize with get_qdrant_store() which is a sync function.

    Pattern: Acquire threading lock to remove from dict, release lock, then
    perform async close operation. This prevents blocking sync callers during
    potentially slow async close operations.

    Args:
        session_id: Specific session to close, or None for all
    """
    async with _get_async_lock():
        if session_id:
            # Extract store under lock, close outside lock
            store_to_close = None
            with _qdrant_lock:
                if session_id in _qdrant_instances:
                    store_to_close = _qdrant_instances.pop(session_id)

            if store_to_close:
                await store_to_close.close()
                logger.info(f"Closed QdrantStore for session: {session_id}")
        else:
            # Extract all stores under lock, close outside lock
            with _qdrant_lock:
                stores_to_close = list(_qdrant_instances.items())
                _qdrant_instances.clear()

            for sid, store in stores_to_close:
                await store.close()
                logger.debug(f"Closed QdrantStore for session: {sid}")
            logger.info("Closed all QdrantStore instances")
