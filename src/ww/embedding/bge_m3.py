"""
BGE-M3 Embedding Provider for World Weaver.

Uses FlagEmbedding library for efficient local inference on RTX 3090.
Supports dense, sparse, and ColBERT embeddings (only dense used for WW).
"""

import heapq
import logging
import threading
from datetime import datetime, timedelta
from hashlib import md5
from typing import Any

import numpy as np

from ww.core.config import get_settings

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Thread-safe cache with TTL eviction.

    Features:
    - Time-based expiration
    - Size-based eviction (oldest entry when full)
    - Thread-safe operations
    - Statistics tracking
    - P2-OPT-B1.1: O(log n) eviction using heap
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize TTL cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[Any, datetime]] = {}
        # P2-OPT-B1.1: Heap for O(log n) eviction instead of O(n) min scan
        self._heap: list[tuple[datetime, str]] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """
        Get value if exists and not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if missing/expired
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.now() - timestamp < self.ttl:
                    self._hits += 1
                    return value
                # Expired - remove it
                del self._cache[key]
            self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set value with current timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()

            now = datetime.now()
            self._cache[key] = (value, now)
            # P2-OPT-B1.1: Push to heap for O(log n) eviction
            heapq.heappush(self._heap, (now, key))

    def _evict_oldest(self) -> None:
        """
        Evict the oldest entry (must hold lock).

        P2-OPT-B1.1 FIX: Uses heap for O(log n) instead of O(n) min scan.
        Handles stale heap entries (keys that were updated or already removed).
        """
        if not self._cache:
            return

        # Pop from heap until we find a valid entry
        while self._heap:
            timestamp, key = heapq.heappop(self._heap)
            if key in self._cache:
                stored_value, stored_timestamp = self._cache[key]
                # Check if this heap entry is current (not stale from an update)
                if stored_timestamp == timestamp:
                    del self._cache[key]
                    return
                # Stale entry - key was updated with new timestamp, skip
            # Key not in cache (already removed) - skip

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._heap.clear()  # P2-OPT-B1.1: Clear heap too
            self._hits = 0
            self._misses = 0

    def evict_expired(self) -> int:
        """
        Evict all expired entries.

        Returns:
            Count of entries evicted
        """
        with self._lock:
            now = datetime.now()
            expired = [
                k for k, (_, ts) in self._cache.items()
                if now - ts >= self.ttl
            ]
            for k in expired:
                del self._cache[k]
            return len(expired)

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl.total_seconds(),
                "heap_size": len(self._heap),  # P2-OPT-B1.1: Track heap size
            }

    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)


class BGEM3Embedding:
    """
    BGE-M3 embedding provider using FlagEmbedding.

    Optimized for RTX 3090 with FP16 inference.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_fp16: bool | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        cache_dir: str | None = None,
        embedding_cache_size: int | None = None,
        embedding_cache_ttl: int | None = None,
    ):
        """
        Initialize BGE-M3 embedding provider.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-m3)
            device: Inference device (default: cuda:0)
            use_fp16: Use FP16 precision (default: True)
            batch_size: Batch size for inference (default: 32)
            max_length: Max token length (default: 512)
            cache_dir: Model cache directory
            embedding_cache_size: Max cache entries (default from config)
            embedding_cache_ttl: Cache TTL in seconds (default from config)
        """
        settings = get_settings()

        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.embedding_use_fp16
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_length = max_length or settings.embedding_max_length
        self.cache_dir = cache_dir or settings.embedding_cache_dir
        self._dimension = settings.embedding_dimension

        # Model lazy loading
        self._model = None
        self._initialized = False
        self._model_lock = threading.Lock()

        # TTL-based embedding cache
        cache_size = embedding_cache_size or settings.embedding_cache_size
        cache_ttl = embedding_cache_ttl or settings.embedding_cache_ttl
        self._cache = TTLCache(max_size=cache_size, ttl_seconds=cache_ttl)

        logger.info(
            f"Initialized BGE-M3 embedding with TTL cache "
            f"(size={cache_size}, ttl={cache_ttl}s)"
        )

    @property
    def dimension(self) -> int:
        """Return embedding dimension (1024 for BGE-M3)."""
        return self._dimension

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key from text using MD5 hash.

        MD5 chosen over full text keys because:
        1. Fixed size (32 chars vs variable length text)
        2. Fast computation (cryptographic strength not needed)
        3. Collision probability negligible for embedding cache use case

        Args:
            text: Input text to hash

        Returns:
            MD5 hash hex digest
        """
        return md5(text.encode(), usedforsecurity=False).hexdigest()

    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of the model with thread-safe double-check locking.

        Thread-safe implementation ensures model is only loaded once even in
        concurrent environments.
        """
        if self._initialized:
            return

        with self._model_lock:
            # Double-check locking pattern
            if self._initialized:
                return

            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info(f"Loading BGE-M3 model: {self.model_name}")
                logger.info(f"Device: {self.device}, FP16: {self.use_fp16}")

                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self.use_fp16,
                    devices=[self.device] if "cuda" in self.device else None,
                    cache_dir=self.cache_dir,
                )

                self._initialized = True
                logger.info("BGE-M3 model loaded successfully")

            except ImportError:
                logger.warning("FlagEmbedding not available, trying sentence-transformers")
                self._init_sentence_transformer()

    def _init_sentence_transformer(self) -> None:
        """Fallback to sentence-transformers if FlagEmbedding unavailable."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading model via sentence-transformers: {self.model_name}")

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_dir,
        )
        self._initialized = True
        self._use_st = True
        logger.info("Model loaded via sentence-transformers")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (1024-dim each)
        """
        self._ensure_initialized()

        if not texts:
            return []

        if hasattr(self, "_use_st") and self._use_st:
            # Sentence-transformers path
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()

        # FlagEmbedding path (returns dict with 'dense_vecs')
        result = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        # Handle both dict and array responses
        if isinstance(result, dict):
            embeddings = result["dense_vecs"]
        else:
            embeddings = result

        # Convert to list if numpy array
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()

        return list(embeddings)

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate query-optimized embedding with caching.

        For BGE-M3, queries and documents use same embedding,
        but we add instruction prefix for better retrieval.

        Uses TTL cache to avoid re-embedding identical queries.

        Args:
            query: Query text

        Returns:
            Embedding vector (1024-dim)
        """
        # BGE-M3 recommends instruction prefix for queries
        instruction = "Represent this sentence for searching relevant passages: "
        prefixed_query = instruction + query

        # Check cache first
        cache_key = self._get_cache_key(prefixed_query)
        cached = self._cache.get(cache_key)

        if cached is not None:
            logger.debug(f"Cache hit for query (hit_rate: {self._cache.stats['hit_rate']:.2%})")
            return cached

        # Cache miss - compute embedding
        logger.debug(f"Cache miss for query (hit_rate: {self._cache.stats['hit_rate']:.2%})")

        embeddings = await self.embed([prefixed_query])
        embedding = embeddings[0]

        # Store in cache (TTL and size eviction handled by TTLCache)
        self._cache.set(cache_key, embedding)

        return embedding

    async def embed_hybrid(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """
        Generate hybrid embeddings (dense + sparse) for texts.

        Uses BGE-M3's native lexical weights for sparse vectors, enabling
        hybrid search that combines semantic similarity with exact matching.

        Args:
            texts: List of texts to embed

        Returns:
            Tuple of (dense_vectors, sparse_vectors) where:
            - dense_vectors: List of 1024-dim float vectors
            - sparse_vectors: List of {token_id: weight} dicts for BM25-like matching
        """
        self._ensure_initialized()

        if not texts:
            return [], []

        # sentence-transformers doesn't support sparse vectors
        if hasattr(self, "_use_st") and self._use_st:
            logger.warning("sentence-transformers fallback doesn't support sparse vectors")
            dense = await self.embed(texts)
            # Return empty sparse vectors as fallback
            return dense, [{} for _ in texts]

        # FlagEmbedding path with sparse support
        result = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # Extract dense vectors
        dense_vecs = result["dense_vecs"]
        if isinstance(dense_vecs, np.ndarray):
            dense_vecs = dense_vecs.tolist()

        # Convert sparse (lexical_weights) to Qdrant format
        sparse_vecs = self._convert_sparse(result.get("lexical_weights", []))

        return dense_vecs, sparse_vecs

    def _convert_sparse(
        self,
        lexical_weights: list,
    ) -> list[dict[int, float]]:
        """
        Convert BGE-M3 lexical weights to Qdrant sparse vector format.

        Args:
            lexical_weights: List of dicts from BGE-M3 with {token_id: weight}

        Returns:
            List of {int: float} dicts compatible with Qdrant SparseVector
        """
        sparse_vectors = []
        for weights in lexical_weights:
            if weights is None:
                sparse_vectors.append({})
                continue
            # Convert keys to int, values to float for Qdrant
            sparse_dict = {int(k): float(v) for k, v in weights.items()}
            sparse_vectors.append(sparse_dict)
        return sparse_vectors

    async def embed_query_hybrid(
        self,
        query: str,
    ) -> tuple[list[float], dict[int, float]]:
        """
        Generate query-optimized hybrid embedding with caching.

        Uses BGE-M3 instruction prefix for better retrieval and caches
        both dense and sparse representations.

        Args:
            query: Query text

        Returns:
            Tuple of (dense_vector, sparse_vector)
        """
        # BGE-M3 recommends instruction prefix for queries
        instruction = "Represent this sentence for searching relevant passages: "
        prefixed_query = instruction + query

        # Check cache (stores tuple of dense, sparse)
        cache_key = "hybrid_" + self._get_cache_key(prefixed_query)
        cached = self._cache.get(cache_key)

        if cached is not None:
            logger.debug(f"Hybrid cache hit for query (hit_rate: {self._cache.stats['hit_rate']:.2%})")
            return cached

        # Cache miss - compute hybrid embedding
        logger.debug(f"Hybrid cache miss for query (hit_rate: {self._cache.stats['hit_rate']:.2%})")

        dense_vecs, sparse_vecs = await self.embed_hybrid([prefixed_query])
        result = (dense_vecs[0], sparse_vecs[0])

        # Store in cache
        self._cache.set(cache_key, result)

        return result

    async def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed large batch with optional progress reporting.

        Args:
            texts: Large list of texts
            show_progress: Show progress bar

        Returns:
            List of embeddings
        """
        self._ensure_initialized()

        if hasattr(self, "_use_st") and self._use_st:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return embeddings.tolist()

        # For FlagEmbedding, process in chunks
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self._model.encode(
                batch,
                batch_size=len(batch),
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

            if isinstance(result, dict):
                embeddings = result["dense_vecs"]
            else:
                embeddings = result

            if isinstance(embeddings, np.ndarray):
                all_embeddings.extend(embeddings.tolist())
            else:
                all_embeddings.extend(list(embeddings))

            if show_progress:
                logger.info(f"Embedded {min(i + self.batch_size, total)}/{total}")

        return all_embeddings

    async def embed_batch_cached(
        self,
        texts: list[str],
        use_query_prefix: bool = False,
    ) -> list[list[float]]:
        """
        Embed batch of texts with caching support.

        P4.3: Reduces redundant embedding computations by checking cache first
        and only computing embeddings for texts not already cached.

        Args:
            texts: List of texts to embed
            use_query_prefix: Whether to add query instruction prefix

        Returns:
            List of embedding vectors (1024-dim each)
        """
        if not texts:
            return []

        # Prepare texts (optionally with prefix)
        instruction = "Represent this sentence for searching relevant passages: "
        processed_texts = [
            instruction + t if use_query_prefix else t
            for t in texts
        ]

        # Check cache for each text
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(processed_texts):
            cache_key = self._get_cache_key(text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Log cache efficiency
        cache_hits = len(texts) - len(uncached_texts)
        if cache_hits > 0:
            logger.debug(
                f"Batch cache: {cache_hits}/{len(texts)} hits "
                f"({100*cache_hits/len(texts):.1f}%)"
            )

        # Embed uncached texts if any
        if uncached_texts:
            new_embeddings = await self.embed(uncached_texts)

            # Store in cache and results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                cache_key = self._get_cache_key(processed_texts[idx])
                self._cache.set(cache_key, embedding)
                results[idx] = embedding

        # All results should now be filled
        return [r for r in results if r is not None]

    def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score [0, 1]
        """
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_cache_stats(self) -> dict:
        """
        Get embedding cache statistics.

        Returns:
            Dictionary containing cache hits, misses, size, hit rate, and TTL
        """
        stats = self._cache.stats
        return {
            **stats,
            "total_requests": stats["hits"] + stats["misses"],
        }

    def clear_cache(self) -> dict:
        """
        Clear the embedding cache and return final statistics.

        Returns:
            Final cache statistics before clearing
        """
        stats = self.get_cache_stats()
        self._cache.clear()
        logger.info(f"Cache cleared. Final stats: {stats}")
        return stats

    def evict_expired(self) -> int:
        """
        Evict expired cache entries.

        Returns:
            Count of entries evicted
        """
        count = self._cache.evict_expired()
        if count > 0:
            logger.debug(f"Evicted {count} expired cache entries")
        return count


# Singleton instance
_embedding_instance: BGEM3Embedding | None = None


def get_embedding_provider() -> BGEM3Embedding:
    """Get or create singleton embedding provider."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = BGEM3Embedding()
    return _embedding_instance
