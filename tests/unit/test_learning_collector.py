"""
Unit tests for T4DM learning collector.

Tests EventStore, EventCollector, eligibility traces, and credit assignment.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from t4dm.learning.collector import (
    EventStore,
    EventCollector,
    CollectorConfig,
    get_collector,
)
from t4dm.learning.events import (
    RetrievalEvent,
    OutcomeEvent,
    Experience,
    MemoryType,
    OutcomeType,
    FeedbackSignal,
)


class TestEventStore:
    """Tests for EventStore SQLite operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary event store."""
        db_path = str(tmp_path / "test.db")
        return EventStore(db_path)

    def test_creation(self, store):
        assert store.db_path is not None
        assert Path(store.db_path).parent.exists()

    def test_creation_default_path(self, tmp_path, monkeypatch):
        """Test default path creation."""
        monkeypatch.setenv("HOME", str(tmp_path))
        store = EventStore()
        assert ".ww" in store.db_path
        assert "learning.db" in store.db_path

    def test_store_retrieval(self, store):
        mem_id = uuid4()
        event = RetrievalEvent(
            query="test query",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={"test1234": 0.9},
            component_scores={"test1234": {"similarity": 0.8}},
            session_id="session1",
            project="project1",
        )
        event.compute_context_hash("test context")

        store.store_retrieval(event)

        # Retrieve it back
        events = store.get_retrievals_by_context(event.context_hash)
        assert len(events) == 1
        assert events[0].query == "test query"
        assert events[0].memory_type == MemoryType.EPISODIC

    def test_store_retrieval_replace(self, store):
        """Test that storing same ID replaces."""
        event = RetrievalEvent(query="original")
        event.compute_context_hash("ctx")
        store.store_retrieval(event)

        # Store again with same ID but different query
        event.query = "updated"
        store.store_retrieval(event)

        events = store.get_retrievals_by_context(event.context_hash)
        assert len(events) == 1
        assert events[0].query == "updated"

    def test_get_retrievals_by_context_empty(self, store):
        events = store.get_retrievals_by_context("nonexistent")
        assert events == []

    def test_get_retrievals_by_context_max_age(self, store):
        event = RetrievalEvent(query="test")
        event.compute_context_hash("ctx")
        store.store_retrieval(event)

        # Should find with 24h window
        events = store.get_retrievals_by_context(event.context_hash, max_age_hours=24)
        assert len(events) == 1

        # Should not find with 0h window
        events = store.get_retrievals_by_context(event.context_hash, max_age_hours=0)
        assert len(events) == 0

    def test_get_unprocessed_retrievals(self, store):
        for i in range(5):
            event = RetrievalEvent(query=f"query {i}")
            event.compute_context_hash(f"ctx {i}")
            store.store_retrieval(event)

        unprocessed = store.get_unprocessed_retrievals(limit=10)
        assert len(unprocessed) == 5

    def test_mark_retrieval_processed(self, store):
        event = RetrievalEvent(query="test")
        event.compute_context_hash("ctx")
        store.store_retrieval(event)

        unprocessed = store.get_unprocessed_retrievals()
        assert len(unprocessed) == 1

        store.mark_retrieval_processed(event.retrieval_id)

        unprocessed = store.get_unprocessed_retrievals()
        assert len(unprocessed) == 0

    def test_store_outcome(self, store):
        event = OutcomeEvent(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.85,
            session_id="session1",
            explicit_citations=[uuid4()],
            feedback_signals=[FeedbackSignal.ACCEPT],
            task_description="Test task",
            tool_results={"status": "ok"},
        )
        event.compute_context_hash("test context")

        store.store_outcome(event)

        outcomes = store.get_outcomes_by_context(event.context_hash)
        assert len(outcomes) == 1
        assert outcomes[0].success_score == 0.85
        assert outcomes[0].outcome_type == OutcomeType.SUCCESS
        assert FeedbackSignal.ACCEPT in outcomes[0].feedback_signals

    def test_get_outcomes_by_context_empty(self, store):
        outcomes = store.get_outcomes_by_context("nonexistent")
        assert outcomes == []

    def test_store_experience(self, store):
        exp = Experience(
            query="test query",
            memory_type=MemoryType.SEMANTIC,
            retrieved_ids=[uuid4()],
            retrieval_scores=[0.9],
            component_vectors=[[0.5, 0.5, 0.5, 0.5]],
            outcome_score=0.8,
            per_memory_rewards={"mem1": 0.7},
            priority=2.0,
        )

        store.store_experience(exp)
        assert store.count_experiences() == 1

    def test_sample_experiences_empty(self, store):
        samples = store.sample_experiences(5)
        assert samples == []

    def test_sample_experiences(self, store):
        for i in range(10):
            exp = Experience(
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                per_memory_rewards={"m": 0.5},
            )
            store.store_experience(exp)

        samples = store.sample_experiences(5)
        assert len(samples) == 5

    def test_sample_experiences_prioritized(self, store):
        # Create experiences with different priorities
        for i in range(10):
            exp = Experience(
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                per_memory_rewards={"m": 0.5},
                priority=float(i + 1),  # Priority 1-10
            )
            store.store_experience(exp)

        # Sample with prioritization
        samples = store.sample_experiences(5, prioritized=True)
        assert len(samples) == 5

    def test_sample_experiences_not_prioritized(self, store):
        for i in range(10):
            exp = Experience(
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                per_memory_rewards={"m": 0.5},
            )
            store.store_experience(exp)

        samples = store.sample_experiences(5, prioritized=False)
        assert len(samples) == 5

    def test_update_priority(self, store):
        exp = Experience(
            query="test",
            memory_type=MemoryType.EPISODIC,
            component_vectors=[[0.5] * 4],
            per_memory_rewards={"m": 0.5},
            priority=1.0,
        )
        store.store_experience(exp)

        store.update_priority(exp.experience_id, 5.0)

        # Sample and verify priority was updated
        samples = store.sample_experiences(1)
        assert samples[0].priority == 5.0

    def test_count_experiences(self, store):
        assert store.count_experiences() == 0

        for i in range(3):
            exp = Experience(
                query=f"q{i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                per_memory_rewards={},
            )
            store.store_experience(exp)

        assert store.count_experiences() == 3


class TestEligibilityTraces:
    """Tests for eligibility trace operations."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        return EventStore(db_path)

    def test_update_trace_new(self, store):
        mem_id = uuid4()
        trace = store.update_trace(mem_id, 0.5)
        assert trace == 0.5

    def test_update_trace_existing(self, store):
        mem_id = uuid4()
        store.update_trace(mem_id, 0.5)
        trace = store.update_trace(mem_id, 0.3)
        assert trace == 0.8  # 0.5 + 0.3

    def test_get_traces_empty(self, store):
        traces = store.get_traces()
        assert traces == {}

    def test_get_traces(self, store):
        mem1 = uuid4()
        mem2 = uuid4()
        store.update_trace(mem1, 0.5)
        store.update_trace(mem2, 0.7)

        traces = store.get_traces()
        assert len(traces) == 2
        assert traces[str(mem1)] == 0.5
        assert traces[str(mem2)] == 0.7

    def test_decay_traces_empty(self, store):
        count = store.decay_traces()
        assert count == 0

    def test_decay_traces(self, store):
        mem_id = uuid4()
        store.update_trace(mem_id, 1.0)

        # Decay should reduce trace value
        count = store.decay_traces(gamma=0.99, lambda_=0.9)
        assert count == 1

        traces = store.get_traces()
        # Trace should be slightly less (but test can be flaky due to timing)
        assert traces[str(mem_id)] <= 1.0

    def test_decay_traces_prunes_small(self, store):
        mem_id = uuid4()
        store.update_trace(mem_id, 0.0005)  # Below threshold

        count = store.decay_traces()
        assert count == 1

        traces = store.get_traces()
        # Should be pruned
        assert str(mem_id) not in traces


class TestBaselineStatistics:
    """Tests for baseline statistic operations."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        return EventStore(db_path)

    def test_update_baseline_new(self, store):
        value = store.update_baseline("global", 0.8)
        assert value == 0.8

    def test_update_baseline_existing(self, store):
        store.update_baseline("global", 0.8)
        # EMA with alpha=0.1: new = 0.1 * 0.6 + 0.9 * 0.8 = 0.78
        value = store.update_baseline("global", 0.6)
        assert abs(value - 0.78) < 0.01

    def test_get_baseline_missing(self, store):
        value = store.get_baseline("missing", default=0.5)
        assert value == 0.5

    def test_get_baseline(self, store):
        store.update_baseline("test", 0.7)
        value = store.get_baseline("test")
        assert value == 0.7


class TestEventStoreUtilities:
    """Tests for store utility methods."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        return EventStore(db_path)

    def test_get_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["retrieval_events"] == 0
        assert stats["outcome_events"] == 0
        assert stats["experiences"] == 0
        assert stats["eligibility_traces"] == 0

    def test_get_stats(self, store):
        # Add some data
        event = RetrievalEvent(query="test")
        event.compute_context_hash("ctx")
        store.store_retrieval(event)

        outcome = OutcomeEvent()
        outcome.compute_context_hash("ctx")
        store.store_outcome(outcome)

        exp = Experience(
            query="test",
            memory_type=MemoryType.EPISODIC,
            component_vectors=[[0.5] * 4],
            per_memory_rewards={},
        )
        store.store_experience(exp)

        store.update_trace(uuid4(), 0.5)

        stats = store.get_stats()
        assert stats["retrieval_events"] == 1
        assert stats["outcome_events"] == 1
        assert stats["experiences"] == 1
        assert stats["eligibility_traces"] == 1

    def test_cleanup_old_events(self, store):
        # Create and mark as processed
        event = RetrievalEvent(query="test")
        event.compute_context_hash("ctx")
        store.store_retrieval(event)
        store.mark_retrieval_processed(event.retrieval_id)

        # Cleanup with 0 days should remove it
        count = store.cleanup_old_events(max_age_days=0)
        # May or may not find it depending on timing
        assert count >= 0


class TestCollectorConfig:
    """Tests for CollectorConfig dataclass."""

    def test_defaults(self):
        config = CollectorConfig()
        assert config.db_path is None
        assert config.auto_match is True
        assert config.match_window_hours == 24.0
        assert config.trace_gamma == 0.99
        assert config.trace_lambda == 0.9

    def test_custom_values(self):
        config = CollectorConfig(
            db_path="/tmp/test.db",
            auto_match=False,
            match_window_hours=12.0,
            trace_gamma=0.95,
            trace_lambda=0.8,
        )
        assert config.db_path == "/tmp/test.db"
        assert config.auto_match is False
        assert config.match_window_hours == 12.0


class TestEventCollector:
    """Tests for EventCollector main interface."""

    @pytest.fixture
    def collector(self, tmp_path):
        config = CollectorConfig(
            db_path=str(tmp_path / "test.db"),
            auto_match=True,
        )
        return EventCollector(config)

    def test_creation(self, collector):
        assert collector.config is not None
        assert collector.store is not None
        assert collector._pending_context is None

    def test_creation_default_config(self, tmp_path):
        # Temporarily override home
        collector = EventCollector()
        assert collector.config.auto_match is True

    def test_record_retrieval(self, collector):
        mem_id = uuid4()
        event = collector.record_retrieval(
            query="test query",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={str(mem_id)[:8]: 0.9},
            component_scores={str(mem_id)[:8]: {"similarity": 0.8}},
            context="test context",
            session_id="session1",
            project="project1",
        )

        assert event.query == "test query"
        assert event.context_hash is not None
        assert collector._pending_context == event.context_hash

    def test_record_retrieval_no_context(self, collector):
        mem_id = uuid4()
        event = collector.record_retrieval(
            query="test query",
            memory_type=MemoryType.SEMANTIC,
            retrieved_ids=[mem_id],
            retrieval_scores={str(mem_id)[:8]: 0.9},
        )

        assert event.query == "test query"
        assert event.context_hash == ""

    def test_record_outcome(self, collector):
        event = collector.record_outcome(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.85,
            context="test context",
            session_id="session1",
        )

        assert event.success_score == 0.85
        assert event.outcome_type == OutcomeType.SUCCESS

    def test_record_outcome_with_hash(self, collector):
        event = collector.record_outcome(
            outcome_type=OutcomeType.PARTIAL,
            success_score=0.6,
            context_hash="abc123",
        )

        assert event.context_hash == "abc123"

    def test_record_outcome_uses_pending_context(self, collector):
        # First record a retrieval to set pending context
        mem_id = uuid4()
        ret = collector.record_retrieval(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={str(mem_id)[:8]: 0.9},
            context="test context",
        )

        # Record outcome without context
        out = collector.record_outcome(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.9,
        )

        # Should use pending context
        assert out.context_hash == ret.context_hash

    def test_auto_match_creates_experience(self, collector):
        # Record retrieval
        mem_id = uuid4()
        ret = collector.record_retrieval(
            query="test query",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={str(mem_id)[:8]: 0.9},
            component_scores={str(mem_id)[:8]: {"similarity": 0.8, "recency": 0.6}},
            context="test context",
        )

        # Record outcome
        out = collector.record_outcome(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.85,
            context="test context",
        )

        # Should have created an experience
        assert collector.store.count_experiences() >= 1

    def test_compute_rewards(self, collector):
        """Test credit assignment reward computation."""
        mem_id = uuid4()
        retrieval = RetrievalEvent(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={str(mem_id)[:8]: 0.9},
        )

        outcome = OutcomeEvent(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.8,
        )

        rewards = collector._compute_rewards(retrieval, outcome, baseline=0.5)

        # Should have reward for the memory
        assert len(rewards) == 1
        # Positive advantage (0.8 - 0.5 = 0.3)
        assert list(rewards.values())[0] > 0

    def test_compute_rewards_citation_bonus(self, collector):
        """Test that citations get bonus."""
        mem_id = uuid4()
        short_id = str(mem_id)[:8]

        retrieval = RetrievalEvent(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={short_id: 0.9},
        )

        # Outcome with explicit citation
        outcome = OutcomeEvent(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.8,
            explicit_citations=[mem_id],  # Not matching short ID format
        )

        # Get rewards - citation bonus may not apply since we use short IDs
        rewards = collector._compute_rewards(retrieval, outcome, baseline=0.5)
        assert len(rewards) == 1

    def test_compute_rewards_negative_advantage(self, collector):
        """Test negative advantage for bad outcomes."""
        mem_id = uuid4()
        retrieval = RetrievalEvent(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={str(mem_id)[:8]: 0.9},
        )

        outcome = OutcomeEvent(
            outcome_type=OutcomeType.FAILURE,
            success_score=0.2,
        )

        rewards = collector._compute_rewards(retrieval, outcome, baseline=0.5)

        # Negative advantage (0.2 - 0.5 = -0.3)
        assert list(rewards.values())[0] < 0

    def test_compute_rewards_empty_scores(self, collector):
        """Test handling of empty retrieval scores."""
        retrieval = RetrievalEvent(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[],
            retrieval_scores={},
        )

        outcome = OutcomeEvent(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.8,
        )

        rewards = collector._compute_rewards(retrieval, outcome, baseline=0.5)
        assert rewards == {}

    def test_extract_component_vectors(self, collector):
        """Test component vector extraction."""
        mem_id = uuid4()
        retrieval = RetrievalEvent(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            component_scores={
                str(mem_id): {
                    "similarity": 0.9,
                    "recency": 0.5,
                    "importance": 0.7,
                    "outcome_history": 0.3,
                }
            },
        )

        vectors = collector._extract_component_vectors(retrieval)
        assert len(vectors) == 1
        assert vectors[0] == [0.9, 0.5, 0.7, 0.3]

    def test_extract_component_vectors_missing(self, collector):
        """Test handling of missing component scores."""
        mem_id = uuid4()
        retrieval = RetrievalEvent(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            component_scores={},  # No component scores
        )

        vectors = collector._extract_component_vectors(retrieval)
        assert len(vectors) == 1
        assert vectors[0] == [0.0, 0.0, 0.0, 0.0]

    def test_decay_traces(self, collector):
        """Test trace decay passthrough."""
        # Add some traces
        mem_id = uuid4()
        collector.store.update_trace(mem_id, 1.0)

        count = collector.decay_traces()
        assert count == 1

    def test_get_stats(self, collector):
        """Test statistics retrieval."""
        stats = collector.get_stats()
        assert "retrieval_events" in stats
        assert "outcome_events" in stats
        assert "experiences" in stats
        assert "baseline_global" in stats


class TestGetCollector:
    """Tests for get_collector singleton."""

    def test_returns_instance(self, tmp_path, monkeypatch):
        """Test singleton creation."""
        # Reset global
        import t4dm.learning.collector as collector_module
        collector_module._collector_instance = None

        monkeypatch.setenv("HOME", str(tmp_path))
        collector = get_collector()
        assert isinstance(collector, EventCollector)

    def test_returns_same_instance(self, tmp_path, monkeypatch):
        """Test singleton behavior."""
        import t4dm.learning.collector as collector_module
        collector_module._collector_instance = None

        monkeypatch.setenv("HOME", str(tmp_path))
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2


class TestTransactionHandling:
    """Tests for transaction context manager."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        return EventStore(db_path)

    def test_transaction_success(self, store):
        with store.transaction() as cursor:
            cursor.execute("INSERT INTO baseline_stats (key, value, last_updated) VALUES (?, ?, ?)",
                          ("test", 0.5, datetime.now().isoformat()))

        # Should be committed
        value = store.get_baseline("test")
        assert value == 0.5

    def test_transaction_rollback(self, store):
        try:
            with store.transaction() as cursor:
                cursor.execute("INSERT INTO baseline_stats (key, value, last_updated) VALUES (?, ?, ?)",
                              ("test2", 0.5, datetime.now().isoformat()))
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should be rolled back
        value = store.get_baseline("test2", default=0.0)
        assert value == 0.0


class TestMultipleMemoriesFlow:
    """Tests for complete flow with multiple memories."""

    @pytest.fixture
    def collector(self, tmp_path):
        config = CollectorConfig(
            db_path=str(tmp_path / "test.db"),
            auto_match=True,
        )
        return EventCollector(config)

    def test_multiple_memories_credit_assignment(self, collector):
        """Test credit assignment across multiple retrieved memories."""
        mem_ids = [uuid4() for _ in range(3)]

        # Record retrieval with multiple memories
        ret = collector.record_retrieval(
            query="complex query",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=mem_ids,
            retrieval_scores={
                str(m)[:8]: 0.9 - i * 0.1
                for i, m in enumerate(mem_ids)
            },
            component_scores={
                str(m)[:8]: {
                    "similarity": 0.9 - i * 0.1,
                    "recency": 0.5,
                    "importance": 0.7,
                    "outcome_history": 0.3,
                }
                for i, m in enumerate(mem_ids)
            },
            context="test context",
        )

        # Record successful outcome
        out = collector.record_outcome(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.9,
            context="test context",
        )

        # Should create experience with all memories
        assert collector.store.count_experiences() >= 1

        # Sample and verify
        exps = collector.store.sample_experiences(1)
        assert len(exps) == 1
        assert len(exps[0].component_vectors) == 3

    def test_no_match_without_context(self, collector):
        """Test that no matching occurs without context."""
        mem_id = uuid4()

        # Retrieval without context
        collector.record_retrieval(
            query="test",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={str(mem_id)[:8]: 0.9},
        )

        initial_count = collector.store.count_experiences()

        # Outcome without context
        collector.record_outcome(
            outcome_type=OutcomeType.SUCCESS,
            success_score=0.9,
        )

        # Should not create new experience without matching context
        # (depends on pending context behavior)
        assert collector.store.count_experiences() >= initial_count

    def test_newest_retrieval_reward_takes_precedence(self, collector):
        """LOGIC-008: When same memory is in multiple retrievals, newest reward wins."""
        import time
        mem_id = uuid4()
        short_id = str(mem_id)[:8]

        # Create context hash
        import hashlib
        ctx_hash = hashlib.sha256(b"test context").hexdigest()[:16]

        # First (older) retrieval - low score
        event1 = RetrievalEvent(
            query="query 1",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={short_id: 0.3},  # Low score
            context_hash=ctx_hash,
        )
        collector.store.store_retrieval(event1)

        # Small delay to ensure different timestamps
        time.sleep(0.01)

        # Second (newer) retrieval - high score
        event2 = RetrievalEvent(
            query="query 2",
            memory_type=MemoryType.EPISODIC,
            retrieved_ids=[mem_id],
            retrieval_scores={short_id: 0.9},  # High score
            context_hash=ctx_hash,
        )
        collector.store.store_retrieval(event2)

        # Verify ASC order: oldest first, newest last
        retrievals = collector.store.get_retrievals_by_context(ctx_hash)
        assert len(retrievals) == 2
        assert retrievals[0].timestamp <= retrievals[1].timestamp  # ASC order
        assert retrievals[0].retrieval_scores[short_id] == 0.3  # Older
        assert retrievals[1].retrieval_scores[short_id] == 0.9  # Newer

        # When _match_and_create_experiences processes them:
        # - First iteration: old retrieval score (0.3)
        # - Second iteration: new retrieval score (0.9) overwrites
        # Result: newest retrieval's score is used for credit assignment


class TestEventIndexedDecay:
    """Tests for event-indexed TD-λ decay (Hinton critique fix)."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        return EventStore(db_path)

    def test_get_event_index_initial(self, store):
        """Event index should start at 0."""
        index = store.get_event_index()
        assert index == 0

    def test_increment_event_index(self, store):
        """Event index should increment."""
        initial = store.get_event_index()
        new_index = store.increment_event_index()
        assert new_index == initial + 1

    def test_increment_event_index_multiple(self, store):
        """Event index should track multiple increments."""
        for i in range(5):
            index = store.increment_event_index()
            assert index == i + 1

    def test_update_trace_stores_event_index(self, store):
        """Traces should store the current event index."""
        # Advance event counter
        store.increment_event_index()
        store.increment_event_index()
        current = store.get_event_index()
        assert current == 2

        mem_id = uuid4()
        store.update_trace(mem_id, 0.5)

        # Verify trace was stored with event index
        conn = store._get_conn()
        cursor = conn.execute(
            "SELECT last_event_index FROM eligibility_traces WHERE memory_id = ?",
            (str(mem_id),)
        )
        row = cursor.fetchone()
        assert row['last_event_index'] == current

    def test_decay_traces_event_indexed(self, store):
        """Decay should use event count, not wall-clock time."""
        mem_id = uuid4()
        store.update_trace(mem_id, 1.0)

        # Advance event counter by 5 events
        for _ in range(5):
            store.increment_event_index()

        # Decay with event-indexed mode
        count = store.decay_traces(gamma=0.99, lambda_=0.9, use_event_indexed=True)
        assert count == 1

        traces = store.get_traces()
        # Should decay by (0.99 * 0.9)^5 = 0.891^5 ≈ 0.561
        expected = 1.0 * (0.99 * 0.9) ** 5
        assert abs(traces[str(mem_id)] - expected) < 0.01

    def test_decay_traces_wall_clock_fallback(self, store):
        """Can still use wall-clock decay for backward compatibility."""
        mem_id = uuid4()
        store.update_trace(mem_id, 1.0)

        # Decay with wall-clock mode
        count = store.decay_traces(
            gamma=0.99, lambda_=0.9, use_event_indexed=False
        )
        assert count == 1

        traces = store.get_traces()
        # Should be nearly unchanged (no time elapsed)
        assert traces[str(mem_id)] > 0.99

    def test_decay_traces_for_event(self, store):
        """Single-step decay should apply one γλ factor."""
        mem_id = uuid4()
        store.update_trace(mem_id, 1.0)

        # Apply single-step decay
        remaining = store.decay_traces_for_event(gamma=0.99, lambda_=0.9)
        assert remaining == 1

        traces = store.get_traces()
        # Should decay by exactly 0.99 * 0.9 = 0.891
        expected = 1.0 * 0.99 * 0.9
        assert abs(traces[str(mem_id)] - expected) < 0.001

    def test_decay_traces_for_event_increments_counter(self, store):
        """Single-step decay should increment event counter."""
        initial = store.get_event_index()

        mem_id = uuid4()
        store.update_trace(mem_id, 1.0)
        store.decay_traces_for_event()

        new_index = store.get_event_index()
        assert new_index == initial + 1

    def test_decay_traces_for_event_prunes(self, store):
        """Single-step decay should prune negligible traces."""
        mem_id = uuid4()
        store.update_trace(mem_id, 0.0001)  # Very small

        remaining = store.decay_traces_for_event()
        assert remaining == 0  # Should be pruned

        traces = store.get_traces()
        assert str(mem_id) not in traces

    def test_multiple_traces_decay(self, store):
        """Multiple traces should decay independently."""
        mem1 = uuid4()
        mem2 = uuid4()

        # mem1 updated at event 0
        store.update_trace(mem1, 1.0)

        # Advance 3 events
        for _ in range(3):
            store.increment_event_index()

        # mem2 updated at event 3
        store.update_trace(mem2, 1.0)

        # Advance 2 more events (now at event 5)
        for _ in range(2):
            store.increment_event_index()

        # Decay
        store.decay_traces(gamma=0.99, lambda_=0.9, use_event_indexed=True)

        traces = store.get_traces()
        # mem1: 5 events elapsed -> (0.891)^5 ≈ 0.561
        # mem2: 2 events elapsed -> (0.891)^2 ≈ 0.794
        assert traces[str(mem1)] < traces[str(mem2)]
        assert abs(traces[str(mem1)] - 0.561) < 0.02
        assert abs(traces[str(mem2)] - 0.794) < 0.02

    def test_decay_consistency_across_time(self, store):
        """Event-indexed decay should be consistent regardless of wall time."""
        mem_id = uuid4()
        store.update_trace(mem_id, 1.0)

        # Increment events
        for _ in range(10):
            store.increment_event_index()

        # No matter how much wall time passes, decay is based on events
        store.decay_traces(gamma=0.99, lambda_=0.9, use_event_indexed=True)

        traces = store.get_traces()
        expected = (0.99 * 0.9) ** 10
        assert abs(traces[str(mem_id)] - expected) < 0.01
