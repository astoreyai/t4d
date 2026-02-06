"""
LongMemEval Benchmark (W5-01).

Compare T4DM against Mem0, Letta, Zep on:
- Session-based memory (needle-in-haystack)
- Long-term retention
- Consolidation effectiveness

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
class BenchmarkResult:
    """Result from a single benchmark test."""

    test_name: str
    system: str
    success: bool
    latency_ms: float
    accuracy: float
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "system": self.system,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "accuracy": self.accuracy,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LongMemEvalConfig:
    """Configuration for LongMemEval benchmark."""

    # Needle in haystack settings
    haystack_sizes: list[int] = field(default_factory=lambda: [100, 500, 1000])
    needle_positions: list[str] = field(default_factory=lambda: ["start", "middle", "end"])

    # Retention settings
    retention_intervals_hours: list[int] = field(default_factory=lambda: [1, 24, 168])  # 1h, 1d, 1w

    # Sessions
    n_sessions: int = 10
    items_per_session: int = 50


class NeedleInHaystackTest:
    """Test: Find specific memory in large context."""

    def __init__(self, config: LongMemEvalConfig):
        self.config = config

    def generate_haystack(self, size: int, needle_pos: str) -> tuple[list[str], str]:
        """Generate haystack with embedded needle.

        Args:
            size: Number of distractor items.
            needle_pos: "start", "middle", or "end".

        Returns:
            Tuple of (all_items, needle_text).
        """
        distractors = [f"Distractor item {i}: random content {np.random.randint(10000)}"
                       for i in range(size)]
        needle = f"SECRET_NEEDLE: The answer is {np.random.randint(1000, 9999)}"

        if needle_pos == "start":
            pos = 0
        elif needle_pos == "end":
            pos = size
        else:  # middle
            pos = size // 2

        items = distractors[:pos] + [needle] + distractors[pos:]
        return items, needle

    def run(self, memory_system: Any) -> list[BenchmarkResult]:
        """Run needle-in-haystack tests.

        Args:
            memory_system: System with store() and search() methods.

        Returns:
            List of benchmark results.
        """
        results = []

        for size in self.config.haystack_sizes:
            for pos in self.config.needle_positions:
                items, needle = self.generate_haystack(size, pos)

                # Store all items
                start_store = time.perf_counter()
                for item in items:
                    memory_system.store(item)
                store_time = (time.perf_counter() - start_store) * 1000

                # Search for needle
                query = "What is the secret answer?"
                start_search = time.perf_counter()
                retrieved = memory_system.search(query, k=1)
                search_time = (time.perf_counter() - start_search) * 1000

                # Check if needle found
                found = any("SECRET_NEEDLE" in str(r) for r in retrieved)

                results.append(BenchmarkResult(
                    test_name=f"needle_haystack_{size}_{pos}",
                    system="t4dm",
                    success=found,
                    latency_ms=search_time,
                    accuracy=1.0 if found else 0.0,
                    details={
                        "haystack_size": size,
                        "needle_position": pos,
                        "store_time_ms": store_time,
                    },
                ))

        return results


class RetentionTest:
    """Test: Long-term memory retention after consolidation."""

    def __init__(self, config: LongMemEvalConfig):
        self.config = config

    def run(self, memory_system: Any) -> list[BenchmarkResult]:
        """Run retention tests.

        Note: In a real benchmark, this would span actual time.
        Here we simulate with consolidation cycles.

        Args:
            memory_system: System with store(), search(), consolidate() methods.

        Returns:
            List of benchmark results.
        """
        results = []

        # Store initial memories
        test_memories = [
            ("capital_france", "The capital of France is Paris."),
            ("recipe_pasta", "To make pasta, boil water, add salt, cook al dente."),
            ("einstein_birthday", "Albert Einstein was born on March 14, 1879."),
        ]

        for mem_id, content in test_memories:
            memory_system.store(content, id=mem_id)

        # Simulate time passing with consolidation
        for interval in self.config.retention_intervals_hours:
            # Trigger consolidation (simulates sleep)
            if hasattr(memory_system, "consolidate"):
                memory_system.consolidate()

            # Test retrieval accuracy
            queries = [
                ("What is the capital of France?", "Paris"),
                ("How do you cook pasta?", "boil"),
                ("When was Einstein born?", "1879"),
            ]

            correct = 0
            for query, expected in queries:
                start = time.perf_counter()
                retrieved = memory_system.search(query, k=3)
                latency = (time.perf_counter() - start) * 1000

                found = any(expected.lower() in str(r).lower() for r in retrieved)
                if found:
                    correct += 1

            accuracy = correct / len(queries)

            results.append(BenchmarkResult(
                test_name=f"retention_{interval}h",
                system="t4dm",
                success=accuracy > 0.8,
                latency_ms=latency,
                accuracy=accuracy,
                details={
                    "simulated_hours": interval,
                    "correct_count": correct,
                    "total_queries": len(queries),
                },
            ))

        return results


class SessionMemoryTest:
    """Test: Cross-session memory continuity."""

    def __init__(self, config: LongMemEvalConfig):
        self.config = config

    def run(self, memory_system: Any) -> list[BenchmarkResult]:
        """Run session memory tests.

        Args:
            memory_system: System with store(), search(), new_session() methods.

        Returns:
            List of benchmark results.
        """
        results = []

        # Create memories across sessions
        session_memories = {}

        for session_idx in range(self.config.n_sessions):
            if hasattr(memory_system, "new_session"):
                memory_system.new_session(f"session_{session_idx}")

            session_memories[session_idx] = []

            for item_idx in range(self.config.items_per_session):
                content = f"Session {session_idx} item {item_idx}: data_{np.random.randint(10000)}"
                memory_system.store(content)
                session_memories[session_idx].append(content)

        # Test cross-session retrieval
        total_found = 0
        total_queries = 0

        for session_idx in range(self.config.n_sessions):
            # Query from different session
            query = f"Session {session_idx} item 0"

            start = time.perf_counter()
            retrieved = memory_system.search(query, k=5)
            latency = (time.perf_counter() - start) * 1000

            # Check if correct session's memory found
            found = any(f"Session {session_idx}" in str(r) for r in retrieved)
            if found:
                total_found += 1
            total_queries += 1

        accuracy = total_found / total_queries

        results.append(BenchmarkResult(
            test_name="cross_session_retrieval",
            system="t4dm",
            success=accuracy > 0.9,
            latency_ms=latency,
            accuracy=accuracy,
            details={
                "n_sessions": self.config.n_sessions,
                "items_per_session": self.config.items_per_session,
                "found_count": total_found,
            },
        ))

        return results


class LongMemEvalBenchmark:
    """Complete LongMemEval benchmark suite."""

    def __init__(self, config: Optional[LongMemEvalConfig] = None):
        self.config = config or LongMemEvalConfig()
        self.tests = [
            NeedleInHaystackTest(self.config),
            RetentionTest(self.config),
            SessionMemoryTest(self.config),
        ]

    def run(self, memory_system: Any) -> dict:
        """Run all benchmark tests.

        Args:
            memory_system: Memory system to benchmark.

        Returns:
            Dictionary with benchmark results.
        """
        all_results = []

        for test in self.tests:
            logger.info(f"Running {test.__class__.__name__}")
            results = test.run(memory_system)
            all_results.extend(results)

        # Compute summary statistics
        n_passed = sum(1 for r in all_results if r.success)
        avg_latency = np.mean([r.latency_ms for r in all_results])
        avg_accuracy = np.mean([r.accuracy for r in all_results])

        return {
            "benchmark": "LongMemEval",
            "system": "t4dm",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(all_results),
                "passed": n_passed,
                "failed": len(all_results) - n_passed,
                "pass_rate": n_passed / len(all_results),
                "avg_latency_ms": avg_latency,
                "avg_accuracy": avg_accuracy,
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
    """Run LongMemEval benchmark."""
    logging.basicConfig(level=logging.INFO)

    # Create mock memory system for testing
    class MockMemorySystem:
        def __init__(self):
            self.memories = []

        def store(self, content: str, id: str = None):
            self.memories.append({"id": id, "content": content})

        def search(self, query: str, k: int = 5):
            # Simple substring matching for mock
            results = [m for m in self.memories if query.lower() in m["content"].lower()]
            return results[:k] if results else self.memories[:k]

        def consolidate(self):
            pass

        def new_session(self, session_id: str):
            pass

    # Run benchmark
    config = LongMemEvalConfig(
        haystack_sizes=[100],  # Reduced for quick test
        n_sessions=3,
        items_per_session=10,
    )
    benchmark = LongMemEvalBenchmark(config)
    memory = MockMemorySystem()

    results = benchmark.run(memory)

    # Print summary
    print("\n=== LongMemEval Results ===")
    print(f"Pass Rate: {results['summary']['pass_rate']:.1%}")
    print(f"Avg Latency: {results['summary']['avg_latency_ms']:.2f}ms")
    print(f"Avg Accuracy: {results['summary']['avg_accuracy']:.2f}")

    # Save results
    output_path = Path("benchmarks/longmemeval/results.json")
    benchmark.save_results(results, output_path)


if __name__ == "__main__":
    main()
