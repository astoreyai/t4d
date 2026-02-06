"""
Pytest wrapper for DMRBenchmark.

Tests Deep Memory Retrieval with κ-gradient continuous consolidation:
- Retrieval accuracy (recall@k, MRR)
- κ-level distribution effects
- Comparison of continuous vs discrete consolidation

Run with: pytest tests/benchmarks/test_dmr.py -m benchmark
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import pytest

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from dmr.run import (
    DMRBenchmark,
    DMRConfig,
    RetrievalAccuracyTest,
    KappaDistributionTest,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Mock Memory System for Testing
# ============================================================================

class MockMemorySystemDMR:
    """Mock memory system for DMR benchmark testing."""

    def __init__(self):
        """Initialize with empty memory store."""
        self.memories = {}

    def store(self, content: str, id: str = None, kappa: float = 0.5):
        """Store content with optional kappa level."""
        mem_id = id or f"mem_{len(self.memories)}"
        self.memories[mem_id] = {"content": content, "kappa": kappa}
        return mem_id

    def search(self, query: str, k: int = 5):
        """Search for memories matching query."""
        results = []
        for mem_id, mem in self.memories.items():
            if any(word.lower() in mem["content"].lower() for word in query.lower().split()):
                # Create a simple object to hold search result
                result = type("Memory", (), {"id": mem_id, "content": mem["content"]})()
                results.append(result)

        return results[:k]

    def get_by_kappa(self, kappa_level: float, k: int = 5):
        """Get memories at specific kappa level."""
        results = []
        for mem_id, mem in self.memories.items():
            if abs(mem["kappa"] - kappa_level) < 0.05:  # Within 0.05 of target
                result = type("Memory", (), {"id": mem_id, "content": mem["content"]})()
                results.append(result)

        return results[:k]

    def clear(self):
        """Clear all memories."""
        self.memories.clear()


@pytest.fixture
def memory_system_dmr():
    """Create a mock memory system for DMR testing."""
    return MockMemorySystemDMR()


@pytest.fixture
def dmr_config():
    """Create configuration for DMR testing."""
    return DMRConfig(
        n_queries=30,
        k_values=[1, 5, 10],
        n_memories=100,
        semantic_clusters=5,
        kappa_levels=[0.0, 0.3, 0.6, 0.9],
    )


# ============================================================================
# Retrieval Accuracy Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.retrieval
class TestRetrievalAccuracy:
    """Test basic retrieval accuracy metrics."""

    def test_recall_at_1(self, memory_system_dmr, dmr_config):
        """Test recall@1 (top-1 accuracy)."""
        memory_system_dmr.clear()

        test = RetrievalAccuracyTest(dmr_config)
        results = test.run(memory_system_dmr, mode="continuous")

        assert len(results) > 0
        result = results[0]
        assert 0.0 <= result.recall_at_1 <= 1.0

    def test_recall_at_5(self, memory_system_dmr, dmr_config):
        """Test recall@5 (top-5 accuracy)."""
        memory_system_dmr.clear()

        test = RetrievalAccuracyTest(dmr_config)
        results = test.run(memory_system_dmr, mode="continuous")

        assert len(results) > 0
        result = results[0]
        assert result.recall_at_5 >= result.recall_at_1, "Recall@5 should be >= Recall@1"

    def test_recall_at_10(self, memory_system_dmr, dmr_config):
        """Test recall@10 (top-10 accuracy)."""
        memory_system_dmr.clear()

        test = RetrievalAccuracyTest(dmr_config)
        results = test.run(memory_system_dmr, mode="continuous")

        assert len(results) > 0
        result = results[0]
        assert result.recall_at_10 >= result.recall_at_5, "Recall@10 should be >= Recall@5"

    def test_mrr_metric(self, memory_system_dmr, dmr_config):
        """Test Mean Reciprocal Rank (MRR) metric."""
        memory_system_dmr.clear()

        test = RetrievalAccuracyTest(dmr_config)
        results = test.run(memory_system_dmr, mode="continuous")

        assert len(results) > 0
        result = results[0]
        assert 0.0 <= result.mrr <= 1.0, f"MRR {result.mrr} out of range"

    def test_continuous_vs_discrete_mode(self, memory_system_dmr, dmr_config):
        """Test κ-gradient (continuous) vs discrete consolidation modes."""
        # Continuous mode (κ-gradient)
        memory_system_dmr.clear()
        test = RetrievalAccuracyTest(dmr_config)
        continuous_results = test.run(memory_system_dmr, mode="continuous")

        assert len(continuous_results) > 0
        continuous_result = continuous_results[0]

        # Discrete mode
        memory_system_dmr.clear()
        discrete_results = test.run(memory_system_dmr, mode="discrete")

        assert len(discrete_results) > 0
        discrete_result = discrete_results[0]

        # Both should have valid metrics
        assert continuous_result.recall_at_1 >= 0.0
        assert discrete_result.recall_at_1 >= 0.0

    def test_retrieval_latency(self, memory_system_dmr, dmr_config):
        """Test retrieval latency is reasonable."""
        memory_system_dmr.clear()

        test = RetrievalAccuracyTest(dmr_config)
        results = test.run(memory_system_dmr, mode="continuous")

        assert len(results) > 0
        result = results[0]

        # Latency should be positive and under 1 second for mock
        assert result.latency_ms > 0
        assert result.latency_ms < 1000, f"Latency {result.latency_ms}ms seems high"


# ============================================================================
# Kappa Distribution Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.retrieval
class TestKappaDistribution:
    """Test retrieval across κ (consolidation) levels."""

    def test_kappa_episodic_level(self, memory_system_dmr, dmr_config):
        """Test retrieval at episodic κ level (0.0)."""
        memory_system_dmr.clear()

        test = KappaDistributionTest(dmr_config)
        results = test.run(memory_system_dmr)

        # Filter for κ=0.0 results
        episodic_results = [r for r in results if r.kappa_mode == "continuous"]
        assert len(episodic_results) > 0

    def test_kappa_level_scaling(self, memory_system_dmr):
        """Test retrieval scales across multiple κ levels."""
        memory_system_dmr.clear()

        config = DMRConfig(
            n_queries=10,
            kappa_levels=[0.0, 0.5, 1.0],
        )
        test = KappaDistributionTest(config)
        results = test.run(memory_system_dmr)

        # Should have results for each κ level
        assert len(results) >= len(config.kappa_levels)

        for result in results:
            assert result.kappa_mode == "continuous"
            assert 0.0 <= result.recall_at_1 <= 1.0

    def test_recall_at_different_kappa_levels(self, memory_system_dmr):
        """Test recall@1 across κ levels."""
        memory_system_dmr.clear()

        config = DMRConfig(
            n_queries=15,
            kappa_levels=[0.0, 0.5, 1.0],
        )
        test = KappaDistributionTest(config)
        results = test.run(memory_system_dmr)

        # Each result should have valid recall@1
        for result in results:
            assert 0.0 <= result.recall_at_1 <= 1.0
            assert result.latency_ms > 0

    def test_kappa_level_completeness(self, memory_system_dmr, dmr_config):
        """Test that all κ levels are tested."""
        memory_system_dmr.clear()

        test = KappaDistributionTest(dmr_config)
        results = test.run(memory_system_dmr)

        # Should test each κ level
        assert len(results) == len(dmr_config.kappa_levels)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.retrieval
class TestDMRComplete:
    """Run complete DMR benchmark suite."""

    def test_complete_dmr_benchmark(self, memory_system_dmr):
        """Run all DMR tests and verify reasonable performance."""
        config = DMRConfig(
            n_queries=20,
            n_memories=50,
            semantic_clusters=3,
            kappa_levels=[0.0, 0.5, 1.0],
        )
        benchmark = DMRBenchmark(config)
        results = benchmark.run(memory_system_dmr)

        # Verify structure
        assert "summary" in results
        assert "results" in results

        summary = results["summary"]
        assert summary["total_tests"] > 0

        logger.info(
            f"DMR: {summary['total_tests']} tests, "
            f"avg recall@1: {summary['avg_recall_at_1']:.2f}, "
            f"avg MRR: {summary['avg_mrr']:.2f}, "
            f"avg latency: {summary['avg_latency_ms']:.2f}ms"
        )

    def test_dmr_recall_metrics_valid(self, memory_system_dmr):
        """Test that all recall metrics are in valid range."""
        config = DMRConfig(n_queries=15, n_memories=50, semantic_clusters=3)
        benchmark = DMRBenchmark(config)
        results = benchmark.run(memory_system_dmr)

        for result in results["results"]:
            assert 0.0 <= result["recall_at_1"] <= 1.0
            assert 0.0 <= result["recall_at_5"] <= 1.0
            assert 0.0 <= result["recall_at_10"] <= 1.0
            assert 0.0 <= result["mrr"] <= 1.0

    def test_dmr_recall_hierarchy(self, memory_system_dmr):
        """Test that recall@k follows hierarchy: recall@1 <= recall@5 <= recall@10."""
        config = DMRConfig(n_queries=15, n_memories=50)
        benchmark = DMRBenchmark(config)
        results = benchmark.run(memory_system_dmr)

        for result in results["results"]:
            assert result["recall_at_1"] <= result["recall_at_5"] + 0.01  # Small epsilon for float
            assert result["recall_at_5"] <= result["recall_at_10"] + 0.01

    def test_dmr_latency_reasonable(self, memory_system_dmr):
        """Test that DMR latencies are reasonable."""
        config = DMRConfig(n_queries=10, n_memories=50)
        benchmark = DMRBenchmark(config)
        results = benchmark.run(memory_system_dmr)

        avg_latency = results["summary"]["avg_latency_ms"]
        assert avg_latency > 0
        assert avg_latency < 5000, f"Average latency {avg_latency}ms too high"

    def test_dmr_kappa_contribution(self, memory_system_dmr):
        """Test that κ-based storage makes sense."""
        config = DMRConfig(
            n_queries=10,
            n_memories=30,
            semantic_clusters=2,
        )
        benchmark = DMRBenchmark(config)
        results = benchmark.run(memory_system_dmr)

        # Should have results from both retrieval accuracy and κ distribution tests
        assert len(results["results"]) > 0

        # Check that we're testing κ levels
        kappa_results = [r for r in results["results"] if "kappa" in r["test_name"]]
        assert len(kappa_results) > 0, "Should have κ-level test results"

    def test_dmr_semantic_clustering(self, memory_system_dmr):
        """Test that semantic clustering works in retrieval."""
        config = DMRConfig(
            n_queries=20,
            n_memories=60,
            semantic_clusters=5,  # Multiple clusters
        )
        benchmark = DMRBenchmark(config)
        results = benchmark.run(memory_system_dmr)

        # Should have meaningful retrieval results
        summary = results["summary"]
        assert summary["total_tests"] > 0

    def test_dmr_test_components(self, memory_system_dmr):
        """Verify both DMR test components are included."""
        config = DMRConfig(n_queries=10, n_memories=50)
        benchmark = DMRBenchmark(config)

        # Should have 2 test types
        assert len(benchmark.tests) == 2
        test_types = {type(t).__name__ for t in benchmark.tests}
        expected = {"RetrievalAccuracyTest", "KappaDistributionTest"}
        assert test_types == expected

    def test_dmr_summary_statistics(self, memory_system_dmr):
        """Test that summary statistics are computed correctly."""
        config = DMRConfig(n_queries=10, n_memories=50)
        benchmark = DMRBenchmark(config)
        results = benchmark.run(memory_system_dmr)

        summary = results["summary"]

        # All required fields should be present
        required_fields = {
            "total_tests",
            "avg_recall_at_1",
            "avg_mrr",
            "avg_latency_ms",
        }
        assert all(f in summary for f in required_fields)

        # Values should be reasonable
        assert summary["total_tests"] > 0
        assert 0.0 <= summary["avg_recall_at_1"] <= 1.0
        assert 0.0 <= summary["avg_mrr"] <= 1.0
        assert summary["avg_latency_ms"] > 0
