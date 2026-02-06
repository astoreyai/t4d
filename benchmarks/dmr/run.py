"""
Deep Memory Retrieval (DMR) Benchmark (W5-02).

Measure retrieval accuracy comparing:
- κ-gradient continuous consolidation
- Discrete memory stores

Evidence Base: Expert recommendations panel benchmarks
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DMRResult:
    """Result from a single DMR test."""

    test_name: str
    kappa_mode: str  # "continuous" or "discrete"
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    latency_ms: float
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "kappa_mode": self.kappa_mode,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DMRConfig:
    """Configuration for DMR benchmark."""

    # Query settings
    n_queries: int = 100
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])

    # Memory distribution
    n_memories: int = 1000
    semantic_clusters: int = 10

    # Kappa distribution test
    kappa_levels: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 0.9])


class RetrievalAccuracyTest:
    """Test: Basic retrieval accuracy metrics."""

    def __init__(self, config: DMRConfig):
        self.config = config

    def compute_metrics(
        self,
        queries: list[tuple[str, str]],
        memory_system: Any,
    ) -> tuple[float, float, float, float, float]:
        """Compute retrieval metrics.

        Args:
            queries: List of (query, expected_id) tuples.
            memory_system: System with search() method.

        Returns:
            Tuple of (recall@1, recall@5, recall@10, MRR, avg_latency_ms).
        """
        recall_1, recall_5, recall_10 = 0, 0, 0
        reciprocal_ranks = []
        latencies = []

        for query, expected_id in queries:
            start = time.perf_counter()
            retrieved = memory_system.search(query, k=10)
            latencies.append((time.perf_counter() - start) * 1000)

            # Find position of expected item
            retrieved_ids = [getattr(r, "id", str(r)) for r in retrieved]

            if expected_id in retrieved_ids:
                rank = retrieved_ids.index(expected_id) + 1
                reciprocal_ranks.append(1.0 / rank)

                if rank <= 1:
                    recall_1 += 1
                if rank <= 5:
                    recall_5 += 1
                if rank <= 10:
                    recall_10 += 1
            else:
                reciprocal_ranks.append(0.0)

        n = len(queries)
        return (
            recall_1 / n,
            recall_5 / n,
            recall_10 / n,
            np.mean(reciprocal_ranks),
            np.mean(latencies),
        )

    def run(self, memory_system: Any, mode: str = "continuous") -> list[DMRResult]:
        """Run retrieval accuracy tests.

        Args:
            memory_system: System with store(), search() methods.
            mode: "continuous" (κ-gradient) or "discrete".

        Returns:
            List of benchmark results.
        """
        results = []

        # Generate clustered test data
        np.random.seed(42)
        memories = []
        queries = []

        for cluster_idx in range(self.config.semantic_clusters):
            # Each cluster has related memories
            cluster_topic = f"topic_{cluster_idx}"

            for item_idx in range(self.config.n_memories // self.config.semantic_clusters):
                mem_id = f"{cluster_topic}_item_{item_idx}"
                content = f"Memory about {cluster_topic}: content_{np.random.randint(10000)}"
                memories.append((mem_id, content, cluster_idx))

            # Add query for this cluster
            query = f"Information about {cluster_topic}"
            expected_id = f"{cluster_topic}_item_0"
            queries.append((query, expected_id))

        # Store memories
        for mem_id, content, cluster_idx in memories:
            if mode == "continuous":
                # κ-gradient: assign kappa based on cluster importance
                kappa = min(1.0, 0.1 + cluster_idx * 0.1)
                memory_system.store(content, id=mem_id, kappa=kappa)
            else:
                # Discrete: just store
                memory_system.store(content, id=mem_id)

        # Compute metrics
        r1, r5, r10, mrr, latency = self.compute_metrics(queries, memory_system)

        results.append(DMRResult(
            test_name="cluster_retrieval",
            kappa_mode=mode,
            recall_at_1=r1,
            recall_at_5=r5,
            recall_at_10=r10,
            mrr=mrr,
            latency_ms=latency,
            details={
                "n_memories": len(memories),
                "n_queries": len(queries),
                "n_clusters": self.config.semantic_clusters,
            },
        ))

        return results


class KappaDistributionTest:
    """Test: Compare retrieval across κ levels."""

    def __init__(self, config: DMRConfig):
        self.config = config

    def run(self, memory_system: Any) -> list[DMRResult]:
        """Run κ-level comparison tests.

        Args:
            memory_system: System with store(), search(), get_by_kappa() methods.

        Returns:
            List of benchmark results.
        """
        results = []

        # Store memories at different κ levels
        np.random.seed(43)
        queries_per_kappa = self.config.n_queries // len(self.config.kappa_levels)

        for kappa in self.config.kappa_levels:
            memories = []
            queries = []

            for i in range(queries_per_kappa * 2):  # 2x for test/query split
                mem_id = f"kappa_{kappa:.1f}_item_{i}"
                content = f"Memory at kappa {kappa:.1f}: content_{np.random.randint(10000)}"
                memories.append((mem_id, content))

            # Store memories
            for mem_id, content in memories:
                memory_system.store(content, id=mem_id, kappa=kappa)

            # Create queries for half
            for i in range(queries_per_kappa):
                query = f"kappa {kappa:.1f} item {i}"
                expected_id = f"kappa_{kappa:.1f}_item_{i}"
                queries.append((query, expected_id))

            # Measure retrieval
            recall_1 = 0
            latencies = []

            for query, expected_id in queries:
                start = time.perf_counter()
                retrieved = memory_system.search(query, k=5)
                latencies.append((time.perf_counter() - start) * 1000)

                retrieved_ids = [getattr(r, "id", str(r)) for r in retrieved]
                if expected_id in retrieved_ids[:1]:
                    recall_1 += 1

            results.append(DMRResult(
                test_name=f"kappa_level_{kappa:.1f}",
                kappa_mode="continuous",
                recall_at_1=recall_1 / len(queries),
                recall_at_5=0.0,  # Not computed for this test
                recall_at_10=0.0,
                mrr=0.0,
                latency_ms=np.mean(latencies),
                details={
                    "kappa_level": kappa,
                    "n_queries": len(queries),
                },
            ))

        return results


class DMRBenchmark:
    """Complete DMR benchmark suite."""

    def __init__(self, config: Optional[DMRConfig] = None):
        self.config = config or DMRConfig()
        self.tests = [
            RetrievalAccuracyTest(self.config),
            KappaDistributionTest(self.config),
        ]

    def run(self, memory_system: Any) -> dict:
        """Run all DMR tests.

        Args:
            memory_system: Memory system to benchmark.

        Returns:
            Dictionary with benchmark results.
        """
        all_results = []

        # Run with continuous κ
        for test in self.tests:
            logger.info(f"Running {test.__class__.__name__}")
            if isinstance(test, RetrievalAccuracyTest):
                results = test.run(memory_system, mode="continuous")
            else:
                results = test.run(memory_system)
            all_results.extend(results)

        # Compute summary
        avg_recall_1 = np.mean([r.recall_at_1 for r in all_results if r.recall_at_1 > 0])
        avg_mrr = np.mean([r.mrr for r in all_results if r.mrr > 0])
        avg_latency = np.mean([r.latency_ms for r in all_results])

        return {
            "benchmark": "DMR",
            "system": "t4dm",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(all_results),
                "avg_recall_at_1": avg_recall_1,
                "avg_mrr": avg_mrr,
                "avg_latency_ms": avg_latency,
            },
            "results": [r.to_dict() for r in all_results],
        }

    def save_results(self, results: dict, output_path: Path) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    """Run DMR benchmark."""
    logging.basicConfig(level=logging.INFO)

    # Create mock memory system
    class MockMemorySystem:
        def __init__(self):
            self.memories = {}

        def store(self, content: str, id: str = None, kappa: float = 0.5):
            self.memories[id] = {"content": content, "kappa": kappa}

        def search(self, query: str, k: int = 5):
            # Return memories matching query substring
            results = []
            for mem_id, mem in self.memories.items():
                if any(word in mem["content"].lower() for word in query.lower().split()):
                    results.append(type("Memory", (), {"id": mem_id, "content": mem["content"]}))
            return results[:k]

    # Run benchmark
    config = DMRConfig(
        n_queries=50,
        n_memories=200,
        semantic_clusters=5,
    )
    benchmark = DMRBenchmark(config)
    memory = MockMemorySystem()

    results = benchmark.run(memory)

    # Print summary
    print("\n=== DMR Results ===")
    print(f"Avg Recall@1: {results['summary']['avg_recall_at_1']:.2f}")
    print(f"Avg MRR: {results['summary']['avg_mrr']:.2f}")
    print(f"Avg Latency: {results['summary']['avg_latency_ms']:.2f}ms")

    # Save results
    output_path = Path("benchmarks/dmr/results.json")
    benchmark.save_results(results, output_path)


if __name__ == "__main__":
    main()
