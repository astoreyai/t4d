"""
Pytest configuration for performance benchmarks.

Provides fixtures and utilities for performance testing and
regression detection.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from datetime import datetime
from typing import AsyncGenerator, Callable, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


# ============================================================================
# Performance Test Markers and Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest for performance testing."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )


# ============================================================================
# Timing and Profiling Fixtures
# ============================================================================

class TimingContext:
    """Context manager for timing measurements."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.duration is None:
            return time.time() - self.start_time if self.start_time else 0
        return self.duration


@pytest.fixture
def timing():
    """Provide timing context manager for benchmarks."""
    return TimingContext


@pytest.fixture
def benchmark_timer():
    """Simple timer for quick benchmarks."""
    def timer(label: str = None):
        start = time.time()

        def elapsed():
            delta = time.time() - start
            if label:
                print(f"{label}: {delta:.3f}s")
            return delta

        return elapsed

    return timer


# ============================================================================
# Performance Test Session Fixtures
# ============================================================================

@pytest.fixture
def benchmark_session_id():
    """Generate unique session ID for benchmark tests."""
    return f"bench-{uuid4().hex[:8]}"


# ============================================================================
# Mock Store Fixtures for Benchmarks
# ============================================================================

@pytest_asyncio.fixture
async def benchmark_embedding_provider():
    """
    Mock embedding provider optimized for benchmarks.

    Returns fast mock with 1024-dim vectors.
    """
    mock = AsyncMock()

    # Fast embedding generation
    async def mock_embed_query(text: str):
        """Generate mock embedding (instant)."""
        return [0.1] * 1024

    async def mock_embed_documents(docs: list[str]):
        """Batch embed documents."""
        return [[0.1 + i / len(docs) for i in range(1024)] for _ in docs]

    mock.embed_query = mock_embed_query
    mock.embed_documents = mock_embed_documents

    return mock


@pytest_asyncio.fixture
async def benchmark_qdrant_store():
    """
    Mock Qdrant store optimized for benchmarks.

    Simulates high-performance vector store behavior.
    """
    mock = AsyncMock()

    # Collections
    mock.episodes_collection = "episodes"
    mock.entities_collection = "entities"
    mock.procedures_collection = "procedures"

    # In-memory storage for benchmarks
    mock._vectors = {}
    mock._call_count = 0
    mock._timing = {}

    async def mock_add(collection, ids, vectors, payloads):
        """Add vectors with timing."""
        mock._call_count += 1
        for id_val in ids:
            mock._vectors[id_val] = {"collection": collection}

    async def mock_search(collection, vector, limit, filter=None):
        """Search with timing."""
        mock._call_count += 1
        # Return mock results up to limit
        results = []
        for i in range(min(limit, 100)):
            results.append((f"id-{i}", 0.95 - i * 0.001, {}))
        return results

    async def mock_get(collection, ids):
        """Get by IDs."""
        mock._call_count += 1
        results = []
        for id_val in ids:
            if id_val in mock._vectors:
                results.append((id_val, {}))
        return results

    async def mock_update_payload(collection, id_val, payload):
        """Update payload."""
        mock._call_count += 1

    async def mock_delete(collection, ids):
        """Delete by IDs."""
        mock._call_count += 1
        for id_val in ids:
            mock._vectors.pop(id_val, None)

    # Assign methods
    mock.initialize = AsyncMock()
    mock.add = mock_add
    mock.search = mock_search
    mock.get = mock_get
    mock.update_payload = mock_update_payload
    mock.delete = mock_delete
    mock.close = AsyncMock()

    return mock


@pytest_asyncio.fixture
async def benchmark_neo4j_store():
    """
    Mock Neo4j store optimized for benchmarks.

    Simulates high-performance graph store behavior.
    """
    mock = AsyncMock()

    # In-memory storage
    mock._nodes = {}
    mock._relationships = {}
    mock._call_count = 0

    async def mock_create_node(label, properties):
        """Create node with timing."""
        mock._call_count += 1
        node_id = f"node-{len(mock._nodes)}"
        mock._nodes[node_id] = {"label": label, "properties": properties}
        return node_id

    async def mock_create_relationship(source_id, target_id, rel_type, properties=None):
        """Create relationship."""
        mock._call_count += 1
        rel_key = f"{source_id}-{rel_type}-{target_id}"
        mock._relationships[rel_key] = {
            "source": source_id,
            "target": target_id,
            "type": rel_type,
        }

    async def mock_get_relationships(source_id, rel_type=None):
        """Get relationships."""
        mock._call_count += 1
        results = []
        for rel in mock._relationships.values():
            if rel["source"] == source_id:
                if rel_type is None or rel["type"] == rel_type:
                    results.append(rel)
        return results[:100]  # Limit for benchmarks

    async def mock_update_node(node_id, properties, label):
        """Update node."""
        mock._call_count += 1
        if node_id in mock._nodes:
            mock._nodes[node_id]["properties"].update(properties)

    async def mock_query(query, **kwargs):
        """Execute query."""
        mock._call_count += 1
        return []

    # Assign methods
    mock.initialize = AsyncMock()
    mock.create_node = mock_create_node
    mock.create_relationship = mock_create_relationship
    mock.get_relationships = mock_get_relationships
    mock.update_node = mock_update_node
    mock.query = mock_query
    mock.close = AsyncMock()

    return mock


# ============================================================================
# Benchmark Result Tracking
# ============================================================================

class BenchmarkResults:
    """Collect and report benchmark results."""

    def __init__(self):
        self.results = []

    def record(
        self,
        name: str,
        duration: float,
        threshold: float,
        unit: str = "s",
        metadata: dict = None,
    ):
        """Record benchmark result."""
        passed = duration < threshold
        self.results.append({
            "name": name,
            "duration": duration,
            "threshold": threshold,
            "unit": unit,
            "passed": passed,
            "metadata": metadata or {},
        })

    def report(self):
        """Print benchmark report."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        passed_count = sum(1 for r in self.results if r["passed"])
        total_count = len(self.results)

        for result in self.results:
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"[{status}] {result['name']:<40} "
                f"{result['duration']:>8.3f}{result['unit']} "
                f"(threshold: {result['threshold']}{result['unit']})"
            )

        print("=" * 70)
        print(f"Passed: {passed_count}/{total_count}")
        print("=" * 70 + "\n")

        return passed_count == total_count


@pytest.fixture
def benchmark_results():
    """Provide benchmark results tracker."""
    return BenchmarkResults()


# ============================================================================
# Async Benchmark Utilities
# ============================================================================

@pytest.fixture
def async_benchmark():
    """Benchmark async functions."""

    async def run(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """
        Run async function and measure time.

        Returns:
            (result, elapsed_time)
        """
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed

    return run


@pytest.fixture
def repeat_benchmark():
    """Benchmark with multiple runs and statistics."""

    def run(
        func: Callable,
        runs: int = 5,
        *args,
        **kwargs
    ) -> dict:
        """
        Run function multiple times and collect statistics.

        Returns:
            {
                'min': float,
                'max': float,
                'mean': float,
                'std': float,
                'samples': list[float],
            }
        """
        samples = []
        for _ in range(runs):
            start = time.time()
            func(*args, **kwargs)
            elapsed = time.time() - start
            samples.append(elapsed)

        return {
            "min": min(samples),
            "max": max(samples),
            "mean": sum(samples) / len(samples),
            "std": (
                (sum((x - sum(samples) / len(samples)) ** 2 for x in samples) / len(samples))
                ** 0.5
            ),
            "samples": samples,
        }

    return run


# ============================================================================
# Performance Assertions
# ============================================================================

def assert_performance(
    actual: float,
    threshold: float,
    metric: str = "execution time",
    unit: str = "s",
):
    """
    Assert that metric is within threshold.

    Args:
        actual: Measured value
        threshold: Maximum acceptable value
        metric: Description of what was measured
        unit: Unit of measurement

    Raises:
        AssertionError if actual > threshold
    """
    assert actual < threshold, (
        f"{metric} exceeded threshold: "
        f"{actual:.3f}{unit} > {threshold:.3f}{unit}"
    )


@pytest.fixture
def assert_perf():
    """Provide performance assertion function."""
    return assert_performance


# ============================================================================
# Load Generation Utilities
# ============================================================================

@pytest.fixture
def generate_load():
    """Generate test load data."""

    class LoadGenerator:
        @staticmethod
        def episodes(count: int, session_id: str = None):
            """Generate episode data."""
            if session_id is None:
                session_id = f"bench-{uuid4().hex[:6]}"

            return [
                {
                    "content": f"Episode {i}: Test event with benchmark content",
                    "context": {"project": "world-weaver", "iteration": i},
                    "outcome": "success" if i % 5 != 0 else "partial",
                    "valence": 0.7 + (i % 10) * 0.01,
                    "session_id": session_id,
                }
                for i in range(count)
            ]

        @staticmethod
        def entities(count: int, session_id: str = None):
            """Generate entity data."""
            if session_id is None:
                session_id = f"bench-{uuid4().hex[:6]}"

            entity_types = ["CONCEPT", "PERSON", "PROJECT", "TOOL", "TECHNIQUE", "FACT"]

            return [
                {
                    "name": f"Entity_{i}",
                    "entity_type": entity_types[i % len(entity_types)],
                    "summary": f"Summary for entity {i}",
                    "details": f"Detailed description for entity {i}" * 5,
                    "session_id": session_id,
                }
                for i in range(count)
            ]

        @staticmethod
        def queries(count: int):
            """Generate search queries."""
            templates = [
                "memory {topic}",
                "{topic} algorithm",
                "how to {topic}",
                "implement {topic}",
                "{topic} performance",
                "{topic} optimization",
                "{topic} best practices",
            ]

            topics = [
                "decay", "consolidation", "storage", "retrieval",
                "learning", "weights", "relationships", "entities",
            ]

            return [
                templates[i % len(templates)].format(topic=topics[i % len(topics)])
                for i in range(count)
            ]

    return LoadGenerator()


# ============================================================================
# Memory Profiling
# ============================================================================

try:
    import psutil

    @pytest.fixture
    def memory_tracker():
        """Track memory usage."""

        class MemoryTracker:
            def __init__(self):
                import os
                self.process = psutil.Process(os.getpid())
                self.baseline = None

            def start(self):
                """Record baseline memory."""
                self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB

            def delta(self):
                """Get memory delta since baseline."""
                if self.baseline is None:
                    return 0
                current = self.process.memory_info().rss / 1024 / 1024
                return current - self.baseline

        return MemoryTracker()

except ImportError:
    pass  # psutil not installed
