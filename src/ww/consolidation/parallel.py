"""
PO-1: Parallel Consolidation Utilities.

Provides multi-process and async parallel execution for consolidation tasks:
- Parallel cluster computation using ProcessPoolExecutor
- Concurrent embedding generation
- Batched async operations with semaphore control

Performance targets:
- 10x faster consolidation for large episode sets
- Non-blocking during CPU-intensive clustering
- Bounded memory usage via chunking
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelConfig:
    """Configuration for parallel consolidation."""

    # Process pool for CPU-bound work (HDBSCAN clustering)
    max_workers: int = 4
    use_process_pool: bool = True

    # Async concurrency limits
    max_concurrent_embeddings: int = 10
    max_concurrent_stores: int = 5

    # Chunking for memory management
    chunk_size: int = 500
    embedding_batch_size: int = 32

    # Timeouts
    cluster_timeout_seconds: float = 300.0
    embedding_timeout_seconds: float = 60.0


@dataclass
class ParallelStats:
    """Statistics from parallel execution."""

    total_items: int = 0
    chunks_processed: int = 0
    parallel_time_seconds: float = 0.0
    sequential_estimate_seconds: float = 0.0
    speedup_factor: float = 1.0
    errors: list[str] = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        """Parallel efficiency (0-1 scale)."""
        if self.sequential_estimate_seconds <= 0:
            return 0.0
        return min(1.0, self.speedup_factor / 4.0)  # Assuming 4 workers


class ParallelExecutor:
    """
    Executor for parallel consolidation operations.

    Provides both process-based parallelism for CPU-bound work
    and async parallelism for I/O-bound operations.
    """

    def __init__(self, config: ParallelConfig | None = None):
        self.config = config or ParallelConfig()
        self._process_pool: ProcessPoolExecutor | None = None
        self._thread_pool: ThreadPoolExecutor | None = None
        self._embedding_semaphore: asyncio.Semaphore | None = None
        self._store_semaphore: asyncio.Semaphore | None = None

    def _get_process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
        return self._process_pool

    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool (fallback for non-picklable functions)."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
        return self._thread_pool

    def _get_embedding_semaphore(self) -> asyncio.Semaphore:
        """Get semaphore for embedding concurrency control."""
        if self._embedding_semaphore is None:
            self._embedding_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_embeddings
            )
        return self._embedding_semaphore

    def _get_store_semaphore(self) -> asyncio.Semaphore:
        """Get semaphore for store operation concurrency control."""
        if self._store_semaphore is None:
            self._store_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_stores
            )
        return self._store_semaphore

    async def parallel_cluster(
        self,
        embeddings_list: list[np.ndarray],
        cluster_fn: Callable[[np.ndarray], np.ndarray],
    ) -> list[np.ndarray]:
        """
        Run clustering in parallel across multiple embedding sets.

        Args:
            embeddings_list: List of embedding arrays to cluster
            cluster_fn: Function that takes embeddings and returns cluster labels

        Returns:
            List of cluster label arrays
        """
        if not embeddings_list:
            return []

        loop = asyncio.get_event_loop()
        pool = self._get_process_pool() if self.config.use_process_pool else self._get_thread_pool()

        # Submit all clustering tasks
        futures = []
        for embeddings in embeddings_list:
            future = loop.run_in_executor(pool, cluster_fn, embeddings)
            futures.append(future)

        # Wait for all with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=self.config.cluster_timeout_seconds,
            )
        except TimeoutError:
            logger.error(f"Parallel clustering timed out after {self.config.cluster_timeout_seconds}s")
            return []

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Clustering chunk {i} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def parallel_embed(
        self,
        texts: list[str],
        embed_fn: Callable[[list[str]], Any],
    ) -> list[np.ndarray]:
        """
        Generate embeddings in parallel batches.

        Args:
            texts: Texts to embed
            embed_fn: Async function that embeds a batch of texts

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        semaphore = self._get_embedding_semaphore()
        batch_size = self.config.embedding_batch_size

        async def embed_batch(batch: list[str]) -> list[np.ndarray]:
            async with semaphore:
                try:
                    return await embed_fn(batch)
                except Exception as e:
                    logger.error(f"Embedding batch failed: {e}")
                    return []

        # Create batches
        batches = [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]

        # Run all batches concurrently
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[embed_batch(batch) for batch in batches]),
                timeout=self.config.embedding_timeout_seconds,
            )
        except TimeoutError:
            logger.error(f"Parallel embedding timed out after {self.config.embedding_timeout_seconds}s")
            return []

        # Flatten results
        embeddings = []
        for batch_result in results:
            embeddings.extend(batch_result)

        return embeddings

    async def parallel_store(
        self,
        items: list[T],
        store_fn: Callable[[T], Any],
    ) -> list[Any]:
        """
        Store items in parallel with concurrency control.

        Args:
            items: Items to store
            store_fn: Async function to store a single item

        Returns:
            List of store results
        """
        if not items:
            return []

        semaphore = self._get_store_semaphore()

        async def store_with_semaphore(item: T) -> Any:
            async with semaphore:
                try:
                    return await store_fn(item)
                except Exception as e:
                    logger.error(f"Store operation failed: {e}")
                    return None

        results = await asyncio.gather(
            *[store_with_semaphore(item) for item in items],
            return_exceptions=True,
        )

        # Filter exceptions
        return [r for r in results if not isinstance(r, Exception)]

    async def chunked_process(
        self,
        items: list[T],
        process_fn: Callable[[list[T]], Any],
        chunk_size: int | None = None,
    ) -> tuple[list[Any], ParallelStats]:
        """
        Process items in chunks with statistics.

        Args:
            items: Items to process
            process_fn: Async function to process a chunk
            chunk_size: Optional override for chunk size

        Returns:
            Tuple of (results, stats)
        """
        import time

        chunk_size = chunk_size or self.config.chunk_size
        stats = ParallelStats(total_items=len(items))

        if not items:
            return [], stats

        start_time = time.time()

        # Create chunks
        chunks = [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]
        stats.chunks_processed = len(chunks)

        # Process chunks concurrently
        results = await asyncio.gather(
            *[process_fn(chunk) for chunk in chunks],
            return_exceptions=True,
        )

        # Collect results and errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                stats.errors.append(f"Chunk {i}: {result}")
            else:
                valid_results.append(result)

        stats.parallel_time_seconds = time.time() - start_time

        # Estimate sequential time (assume linear scaling)
        if stats.chunks_processed > 0 and stats.parallel_time_seconds > 0:
            time_per_chunk = stats.parallel_time_seconds / max(1, len(valid_results))
            stats.sequential_estimate_seconds = time_per_chunk * stats.chunks_processed
            stats.speedup_factor = stats.sequential_estimate_seconds / stats.parallel_time_seconds

        return valid_results, stats

    def shutdown(self) -> None:
        """Shutdown executor pools."""
        if self._process_pool:
            self._process_pool.shutdown(wait=False)
            self._process_pool = None
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None

    def __del__(self):
        self.shutdown()


# Module-level executor instance
_executor: ParallelExecutor | None = None


def get_parallel_executor(config: ParallelConfig | None = None) -> ParallelExecutor:
    """Get or create the parallel executor singleton."""
    global _executor
    if _executor is None:
        _executor = ParallelExecutor(config)
    return _executor


def reset_parallel_executor() -> None:
    """Reset the parallel executor singleton."""
    global _executor
    if _executor:
        _executor.shutdown()
    _executor = None


# Standalone clustering function for process pool (must be picklable)
def cluster_embeddings_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN (process-pool safe).

    This function is designed to be called from a ProcessPoolExecutor.

    Args:
        embeddings: Embedding array (n_samples, n_features)
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        metric: Distance metric

    Returns:
        Cluster labels array
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        logger.error("HDBSCAN not available")
        return np.array([-1] * len(embeddings))

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric=metric,
        cluster_selection_method="eom",
    )

    return clusterer.fit_predict(embeddings)
