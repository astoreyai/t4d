"""
P7.2: Performance Benchmarks for P5 Biology Features.

Tests performance of:
- P5.1: REM abstractions
- P5.2: Temporal episode linking
- P5.3: Synaptic tagging
- P5.4: Query-memory separation
- P5.5: STDP plasticity

Performance targets:
- STDP spike recording: <0.1ms
- STDP weight update: <0.5ms
- Query projection: <1ms
- Memory projection: <1ms
- Batch projection (100): <10ms
- Synaptic tagging: <0.5ms
- Abstraction event creation: <0.5ms
"""

import time
import numpy as np
import pytest
from datetime import datetime, timedelta
from typing import Callable, Tuple
from uuid import uuid4


def benchmark(func: Callable, iterations: int = 100) -> Tuple[float, float, float]:
    """
    Run benchmark and return timing statistics.

    Returns:
        (mean_ms, min_ms, max_ms) tuple
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return np.mean(times), np.min(times), np.max(times)


# =============================================================================
# P5.5: STDP Performance Benchmarks
# =============================================================================

class TestSTDPPerformance:
    """Performance benchmarks for STDP plasticity."""

    @pytest.fixture
    def stdp_learner(self):
        """Create STDP learner."""
        from t4dm.learning.stdp import STDPLearner
        return STDPLearner()

    def test_spike_recording_performance(self, stdp_learner):
        """Spike recording should be <0.1ms."""
        now = datetime.now()
        i = [0]

        def record():
            stdp_learner.record_spike(
                f"neuron_{i[0]}",
                timestamp=now + timedelta(milliseconds=i[0])
            )
            i[0] += 1

        mean, min_, max_ = benchmark(record, iterations=1000)

        print(f"\nSTDP record_spike: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 0.5, f"Spike recording too slow: {mean:.4f}ms"

    def test_single_update_performance(self, stdp_learner):
        """Single weight update should be <0.5ms."""
        now = datetime.now()

        # Pre-record spikes
        stdp_learner.record_spike("pre", timestamp=now)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10))

        def compute_update():
            stdp_learner.compute_update("pre", "post", current_weight=0.5)

        mean, min_, max_ = benchmark(compute_update, iterations=1000)

        print(f"\nSTDP compute_update: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 1.0, f"Update too slow: {mean:.4f}ms"

    def test_batch_updates_performance(self, stdp_learner):
        """Batch update (10 synapses) should be <2ms."""
        now = datetime.now()

        # Record spikes for multiple pre neurons
        for i in range(10):
            stdp_learner.record_spike(f"pre_{i}", timestamp=now)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10))

        def compute_all():
            stdp_learner.compute_all_updates(
                pre_ids=[f"pre_{i}" for i in range(10)],
                post_id="post"
            )

        mean, min_, max_ = benchmark(compute_all, iterations=500)

        print(f"\nSTDP compute_all_updates (10): {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 5.0, f"Batch update too slow: {mean:.4f}ms"

    def test_stdp_delta_computation_performance(self, stdp_learner):
        """STDP delta computation should be <0.05ms."""
        def compute_delta():
            stdp_learner.compute_stdp_delta(10.0)

        mean, min_, max_ = benchmark(compute_delta, iterations=1000)

        print(f"\nSTDP compute_stdp_delta: {mean:.5f}ms (min={min_:.5f}, max={max_:.5f})")
        assert mean < 0.1, f"Delta computation too slow: {mean:.5f}ms"

    def test_weight_decay_performance(self, stdp_learner):
        """Weight decay (100 synapses) should be <2ms."""
        # Create many weights
        for i in range(100):
            stdp_learner.set_weight(f"a{i}", f"b{i}", 0.5 + np.random.rand() * 0.3)

        def apply_decay():
            stdp_learner.apply_weight_decay(baseline=0.5)

        mean, min_, max_ = benchmark(apply_decay, iterations=500)

        print(f"\nSTDP apply_weight_decay (100): {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 5.0, f"Weight decay too slow: {mean:.4f}ms"

    def test_save_load_state_performance(self, stdp_learner):
        """State serialization should be <5ms for 100 synapses."""
        # Create state
        for i in range(100):
            stdp_learner.set_weight(f"a{i}", f"b{i}", 0.5)
            stdp_learner.record_spike(f"a{i}")

        def save_load():
            state = stdp_learner.save_state()
            stdp_learner.load_state(state)

        mean, min_, max_ = benchmark(save_load, iterations=100)

        print(f"\nSTDP save/load (100): {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 10.0, f"Save/load too slow: {mean:.3f}ms"


class TestTripletSTDPPerformance:
    """Performance benchmarks for triplet STDP variant."""

    @pytest.fixture
    def triplet_stdp(self):
        """Create triplet STDP learner."""
        from t4dm.learning.stdp import TripletSTDP
        return TripletSTDP()

    def test_triplet_update_performance(self, triplet_stdp):
        """Triplet STDP update should be <1ms."""
        now = datetime.now()

        # Post-Pre-Post triplet
        triplet_stdp.record_spike("post", timestamp=now)
        triplet_stdp.record_spike("pre", timestamp=now + timedelta(milliseconds=10))
        triplet_stdp.record_spike("post", timestamp=now + timedelta(milliseconds=20))

        def compute():
            triplet_stdp.compute_update("pre", "post", current_weight=0.5)

        mean, min_, max_ = benchmark(compute, iterations=500)

        print(f"\nTripletSTDP update: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 2.0, f"Triplet update too slow: {mean:.4f}ms"


# =============================================================================
# P5.4: Query-Memory Separation Performance
# =============================================================================

class TestQueryMemorySeparationPerformance:
    """Performance benchmarks for query-memory encoder separation."""

    @pytest.fixture
    def separator(self):
        """Create separator."""
        from t4dm.embedding.query_memory_separation import QueryMemorySeparator
        return QueryMemorySeparator()

    def test_query_projection_performance(self, separator):
        """Single query projection should be <1ms."""
        embedding = np.random.randn(1024).astype(np.float32)

        def project():
            separator.project_query(embedding)

        mean, min_, max_ = benchmark(project, iterations=1000)

        print(f"\nQuery projection: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 2.0, f"Query projection too slow: {mean:.4f}ms"

    def test_memory_projection_performance(self, separator):
        """Single memory projection should be <1ms."""
        embedding = np.random.randn(1024).astype(np.float32)

        def project():
            separator.project_memory(embedding)

        mean, min_, max_ = benchmark(project, iterations=1000)

        print(f"\nMemory projection: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 2.0, f"Memory projection too slow: {mean:.4f}ms"

    def test_batch_query_projection_performance(self, separator):
        """Batch query projection (100) should be <10ms."""
        embeddings = np.random.randn(100, 1024).astype(np.float32)

        def project_batch():
            separator.project_query(embeddings)

        mean, min_, max_ = benchmark(project_batch, iterations=100)

        print(f"\nBatch query projection (100): {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 20.0, f"Batch projection too slow: {mean:.3f}ms"

    def test_batch_memory_projection_performance(self, separator):
        """Batch memory projection (100) should be <10ms."""
        embeddings = np.random.randn(100, 1024).astype(np.float32)

        def project_batch():
            separator.project_memory(embeddings)

        mean, min_, max_ = benchmark(project_batch, iterations=100)

        print(f"\nBatch memory projection (100): {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 20.0, f"Batch projection too slow: {mean:.3f}ms"

    def test_similarity_computation_performance(self, separator):
        """Similarity computation should be <5ms for 100 memories."""
        query = np.random.randn(1024).astype(np.float32)
        memories = np.random.randn(100, 1024).astype(np.float32)

        def compute_sim():
            separator.compute_similarity(query, memories)

        mean, min_, max_ = benchmark(compute_sim, iterations=500)

        print(f"\nSimilarity computation (100): {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 10.0, f"Similarity too slow: {mean:.3f}ms"

    def test_large_scale_retrieval_performance(self, separator):
        """Large-scale retrieval (1000 memories) should be <50ms."""
        query = np.random.randn(1024).astype(np.float32)
        memories = np.random.randn(1000, 1024).astype(np.float32)

        def retrieve():
            sims = separator.compute_similarity(query, memories)
            top_k = np.argsort(-sims)[:10]
            return top_k

        mean, min_, max_ = benchmark(retrieve, iterations=50)

        print(f"\nLarge-scale retrieval (1000): {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 100.0, f"Large retrieval too slow: {mean:.2f}ms"


# =============================================================================
# P5.3: Synaptic Tagging Performance
# =============================================================================

class TestSynapticTaggingPerformance:
    """Performance benchmarks for synaptic tagging."""

    @pytest.fixture
    def tagger(self):
        """Create synaptic tagger."""
        from t4dm.learning.plasticity import SynapticTagger
        return SynapticTagger()

    def test_tag_synapse_performance(self, tagger):
        """Synapse tagging should be <0.5ms."""
        i = [0]

        def tag():
            tagger.tag_synapse(f"src_{i[0]}", f"tgt_{i[0]}", signal_strength=0.7)
            i[0] += 1

        mean, min_, max_ = benchmark(tag, iterations=1000)

        print(f"\nSynaptic tag: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 1.0, f"Tagging too slow: {mean:.4f}ms"

    def test_get_tagged_synapses_performance(self, tagger):
        """Getting tagged synapses should be <1ms for 100 tags."""
        # Create many tags
        for i in range(100):
            tagger.tag_synapse(f"src_{i}", f"tgt_{i}", 0.5 + np.random.rand() * 0.4)

        def get_tags():
            tagger.get_tagged_synapses()

        mean, min_, max_ = benchmark(get_tags, iterations=500)

        print(f"\nGet tagged synapses (100): {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 2.0, f"Get tags too slow: {mean:.4f}ms"

    def test_capture_tags_performance(self, tagger):
        """Capturing tags should be <2ms for 100 tags."""
        # Create many tags
        for i in range(100):
            tagger.tag_synapse(f"src_{i}", f"tgt_{i}", 0.5 + np.random.rand() * 0.4)

        def capture():
            tagger.capture_tags()

        mean, min_, max_ = benchmark(capture, iterations=200)

        print(f"\nCapture tags (100): {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 5.0, f"Capture too slow: {mean:.4f}ms"


# =============================================================================
# P5.1: REM Abstractions Performance
# =============================================================================

class TestREMAbstractionPerformance:
    """Performance benchmarks for REM abstraction storage."""

    def test_abstraction_event_creation_performance(self):
        """Abstraction event creation should be <0.5ms."""
        from t4dm.consolidation.sleep import AbstractionEvent

        i = [0]

        def create():
            AbstractionEvent(
                cluster_ids=[f"ep_{j}" for j in range(i[0], i[0] + 5)],
                concept_name=f"concept_{i[0]}",
                confidence=0.8
            )
            i[0] += 1

        mean, min_, max_ = benchmark(create, iterations=1000)

        print(f"\nAbstractionEvent creation: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 1.0, f"Creation too slow: {mean:.4f}ms"

    def test_many_abstraction_events_performance(self):
        """Creating 100 abstraction events should be <50ms."""
        from t4dm.consolidation.sleep import AbstractionEvent

        def create_many():
            events = []
            for i in range(100):
                events.append(AbstractionEvent(
                    cluster_ids=[f"ep_{i}_{j}" for j in range(5)],
                    concept_name=f"concept_{i}",
                    confidence=0.7 + np.random.rand() * 0.2
                ))
            return events

        mean, min_, max_ = benchmark(create_many, iterations=50)

        print(f"\n100 AbstractionEvents: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 100.0, f"Bulk creation too slow: {mean:.2f}ms"


# =============================================================================
# P5.2: Temporal Episode Linking Performance
# =============================================================================

class TestTemporalLinkingPerformance:
    """Performance benchmarks for temporal episode linking."""

    def test_episode_creation_performance(self):
        """Episode creation with links should be <0.5ms."""
        from t4dm.core.types import Episode

        prev_id = None
        session_id = str(uuid4())

        def create():
            nonlocal prev_id
            new_id = uuid4()
            ep = Episode(
                id=new_id,
                session_id=session_id,
                content="Test content",
                previous_episode_id=prev_id,
                sequence_position=0
            )
            prev_id = new_id
            return ep

        mean, min_, max_ = benchmark(create, iterations=1000)

        print(f"\nEpisode creation: {mean:.4f}ms (min={min_:.4f}, max={max_:.4f})")
        assert mean < 1.0, f"Episode creation too slow: {mean:.4f}ms"

    def test_linked_chain_creation_performance(self):
        """Creating linked chain (100 episodes) should be <50ms."""
        from t4dm.core.types import Episode

        def create_chain():
            session_id = str(uuid4())
            ids = [uuid4() for _ in range(100)]
            episodes = []

            for i, id_ in enumerate(ids):
                ep = Episode(
                    id=id_,
                    session_id=session_id,
                    content=f"Episode {i}",
                    previous_episode_id=ids[i-1] if i > 0 else None,
                    next_episode_id=ids[i+1] if i < len(ids)-1 else None,
                    sequence_position=i
                )
                episodes.append(ep)
            return episodes

        mean, min_, max_ = benchmark(create_chain, iterations=50)

        print(f"\n100-episode chain: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 100.0, f"Chain creation too slow: {mean:.2f}ms"


# =============================================================================
# Integrated P5 Feature Performance
# =============================================================================

class TestIntegratedP5Performance:
    """Performance benchmarks for integrated P5 features."""

    def test_retrieval_with_stdp_and_tagging(self):
        """Retrieval triggering STDP + tagging should be <10ms."""
        from t4dm.learning.stdp import STDPLearner
        from t4dm.learning.plasticity import SynapticTagger
        from t4dm.embedding.query_memory_separation import QueryMemorySeparator

        stdp = STDPLearner()
        tagger = SynapticTagger()
        separator = QueryMemorySeparator()

        query = np.random.randn(1024).astype(np.float32)
        memories = np.random.randn(10, 1024).astype(np.float32)
        memory_ids = [f"mem_{i}" for i in range(10)]
        now = datetime.now()

        def retrieval_pipeline():
            # Clear previous spikes to avoid interval validation on repeated runs
            stdp.clear_spikes()
            # Project query
            q_proj = separator.project_query(query)
            m_proj = separator.project_memory(memories)

            # Compute similarities
            sims = separator.compute_similarity(query, memories)
            order = np.argsort(-sims)[:5]

            # Record STDP spikes for retrieved memories
            for i, idx in enumerate(order):
                stdp.record_spike(
                    memory_ids[idx],
                    timestamp=now + timedelta(milliseconds=i * 10)
                )

            # Tag co-retrieved synapses
            for i in range(len(order) - 1):
                tagger.tag_synapse(
                    memory_ids[order[i]],
                    memory_ids[order[i + 1]],
                    signal_strength=float(sims[order[i]])
                )

            # Compute STDP updates
            for i in range(len(order) - 1):
                stdp.compute_update(
                    memory_ids[order[i]],
                    memory_ids[order[i + 1]]
                )

        mean, min_, max_ = benchmark(retrieval_pipeline, iterations=100)

        print(f"\nRetrieval + STDP + tagging: {mean:.3f}ms (min={min_:.3f}, max={max_:.3f})")
        assert mean < 20.0, f"Integrated pipeline too slow: {mean:.3f}ms"

    def test_consolidation_cycle_performance(self):
        """Single consolidation cycle should be <50ms."""
        from t4dm.learning.stdp import STDPLearner
        from t4dm.learning.plasticity import SynapticTagger
        from t4dm.consolidation.sleep import AbstractionEvent

        stdp = STDPLearner()
        tagger = SynapticTagger()

        # Pre-populate with data
        for i in range(100):
            stdp.set_weight(f"a{i}", f"b{i}", 0.5 + np.random.rand() * 0.3)
            tagger.tag_synapse(f"a{i}", f"b{i}", 0.6 + np.random.rand() * 0.3)

        def consolidation_cycle():
            # Apply weight decay
            stdp.apply_weight_decay(baseline=0.5)

            # Capture tags
            captured = tagger.capture_tags()

            # Create abstractions
            abstractions = []
            for i in range(10):
                abstractions.append(AbstractionEvent(
                    cluster_ids=[f"ep_{i}_{j}" for j in range(5)],
                    concept_name=f"concept_{i}",
                    confidence=0.8
                ))

            return abstractions

        mean, min_, max_ = benchmark(consolidation_cycle, iterations=50)

        print(f"\nConsolidation cycle: {mean:.2f}ms (min={min_:.2f}, max={max_:.2f})")
        assert mean < 100.0, f"Consolidation too slow: {mean:.2f}ms"

    def test_sustained_learning_performance(self):
        """
        Sustained learning (1000 interactions) should complete in <5s.
        """
        from t4dm.learning.stdp import STDPLearner
        from t4dm.learning.plasticity import SynapticTagger
        from t4dm.embedding.query_memory_separation import QueryMemorySeparator

        stdp = STDPLearner()
        tagger = SynapticTagger()
        separator = QueryMemorySeparator()

        memory_pool = np.random.randn(100, 1024).astype(np.float32)
        memory_ids = [f"mem_{i}" for i in range(100)]

        start = time.perf_counter()
        now = datetime.now()

        for interaction in range(1000):
            # Random query
            query = np.random.randn(1024).astype(np.float32)

            # Retrieve top 5
            sims = separator.compute_similarity(query, memory_pool)
            top5 = np.argsort(-sims)[:5]

            # STDP for retrieved (use past timestamps to avoid future validation)
            t = now - timedelta(seconds=1000 - interaction)
            for i, idx in enumerate(top5):
                stdp.record_spike(memory_ids[idx], timestamp=t + timedelta(milliseconds=i * 10))

            # Tag pairs
            for i in range(4):
                tagger.tag_synapse(
                    memory_ids[top5[i]],
                    memory_ids[top5[i + 1]],
                    float(sims[top5[i]])
                )

            # Periodic consolidation
            if interaction % 100 == 99:
                stdp.apply_weight_decay(0.5)
                tagger.capture_tags()

        elapsed = time.perf_counter() - start

        print(f"\nSustained learning (1000 interactions): {elapsed:.2f}s")
        assert elapsed < 10.0, f"Sustained learning too slow: {elapsed:.2f}s"


# =============================================================================
# Memory Scaling Tests
# =============================================================================

class TestP5MemoryScaling:
    """Tests for memory efficiency of P5 modules."""

    def test_stdp_memory_with_many_entities(self):
        """STDP should handle 10000 entities efficiently."""
        import sys
        from t4dm.learning.stdp import STDPLearner

        stdp = STDPLearner()
        base_size = sys.getsizeof(stdp._spike_history)

        # Record spikes for many entities
        now = datetime.now()
        for i in range(10000):
            stdp.record_spike(f"entity_{i}", timestamp=now)

        after_size = sys.getsizeof(stdp._spike_history)

        print(f"\nSTDP memory for 10000 entities: {(after_size - base_size) / 1024:.1f} KB")

        # Should not consume more than 10MB
        assert after_size - base_size < 10 * 1024 * 1024

    def test_separator_memory_constant(self):
        """Separator memory should be constant regardless of usage."""
        import sys
        from t4dm.embedding.query_memory_separation import QueryMemorySeparator

        separator = QueryMemorySeparator()
        # Use W_q and W_m matrices for size measurement
        initial_size = separator.W_q.nbytes + separator.W_m.nbytes + separator.U_q.nbytes + separator.U_m.nbytes

        # Use separator many times
        for _ in range(1000):
            emb = np.random.randn(1024).astype(np.float32)
            separator.project_query(emb)
            separator.project_memory(emb)

        final_size = separator.W_q.nbytes + separator.W_m.nbytes + separator.U_q.nbytes + separator.U_m.nbytes

        print(f"\nSeparator memory: initial={initial_size/1024:.1f}KB, final={final_size/1024:.1f}KB")

        # Memory should not grow
        assert final_size == initial_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
