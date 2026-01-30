"""
Tests for Phase 1C: Retrieval Feedback Loop System.

Tests the implicit feedback collection and learning signal processing
that enables continuous learning from user interactions.
"""

import numpy as np
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from ww.learning.retrieval_feedback import (
    RetrievalFeedback,
    RetrievalOutcome,
    RetrievalFeedbackCollector,
)
from ww.learning.feedback_signals import (
    LearningSignal,
    FeedbackSignalProcessor,
    AdapterTrainingSignal,
    FeedbackToAdapterBridge,
)


class TestRetrievalFeedback:
    """Tests for RetrievalFeedback dataclass."""

    def test_creation_defaults(self):
        """Feedback created with default values."""
        fb = RetrievalFeedback()

        assert fb.query_id == ""
        assert fb.result_id == ""
        assert fb.relevance == 0.0
        assert fb.clicked is False
        assert fb.dwell_time == 0.0
        assert fb.confidence == 0.5
        assert isinstance(fb.timestamp, datetime)

    def test_creation_with_values(self):
        """Feedback created with specific values."""
        fb = RetrievalFeedback(
            query_id="q123",
            result_id="r456",
            relevance=0.8,
            clicked=True,
            dwell_time=45.0,
            position=2,
            confidence=0.9,
        )

        assert fb.query_id == "q123"
        assert fb.result_id == "r456"
        assert fb.relevance == 0.8
        assert fb.clicked is True
        assert fb.dwell_time == 45.0
        assert fb.position == 2
        assert fb.confidence == 0.9

    def test_to_dict(self):
        """Feedback serializes to dictionary."""
        fb = RetrievalFeedback(
            query_id="q123",
            result_id="r456",
            relevance=0.8,
            clicked=True,
        )

        d = fb.to_dict()

        assert d["query_id"] == "q123"
        assert d["result_id"] == "r456"
        assert d["relevance"] == 0.8
        assert d["clicked"] is True
        assert "feedback_id" in d
        assert "timestamp" in d

    def test_from_dict(self):
        """Feedback deserializes from dictionary."""
        original = RetrievalFeedback(
            query_id="q123",
            result_id="r456",
            relevance=0.8,
            clicked=True,
            dwell_time=30.0,
        )

        d = original.to_dict()
        restored = RetrievalFeedback.from_dict(d)

        assert restored.query_id == original.query_id
        assert restored.result_id == original.result_id
        assert restored.relevance == original.relevance
        assert restored.clicked == original.clicked
        assert restored.dwell_time == original.dwell_time


class TestRetrievalFeedbackCollector:
    """Tests for RetrievalFeedbackCollector."""

    def test_initialization_defaults(self):
        """Collector initializes with default parameters."""
        collector = RetrievalFeedbackCollector()

        assert collector.click_weight > 0
        assert collector.dwell_weight > 0
        assert collector.position_weight > 0
        assert len(collector._pending) == 0

    def test_initialization_custom_weights(self):
        """Collector normalizes custom weights."""
        collector = RetrievalFeedbackCollector(
            click_weight=0.8,
            dwell_weight=0.2,
            position_weight=0.0,
        )

        # Weights should be normalized
        total = collector.click_weight + collector.dwell_weight + collector.position_weight
        assert abs(total - 1.0) < 0.001

    def test_invalid_weights_raises(self):
        """Invalid weights raise error."""
        with pytest.raises(ValueError):
            RetrievalFeedbackCollector(
                click_weight=0.0,
                dwell_weight=0.0,
                position_weight=0.0,
            )

    def test_start_retrieval(self):
        """Start retrieval creates pending outcome."""
        collector = RetrievalFeedbackCollector()

        query_id = collector.start_retrieval(
            results=["r1", "r2", "r3"],
            scores={"r1": 0.9, "r2": 0.7, "r3": 0.5},
        )

        assert query_id is not None
        assert len(query_id) > 0
        assert query_id in collector._pending
        assert collector._pending[query_id].result_ids == ["r1", "r2", "r3"]

    def test_start_retrieval_custom_id(self):
        """Start retrieval accepts custom query ID."""
        collector = RetrievalFeedbackCollector()

        query_id = collector.start_retrieval(
            results=["r1"],
            query_id="custom_query_123",
        )

        assert query_id == "custom_query_123"

    def test_record_click(self):
        """Click is recorded for pending retrieval."""
        collector = RetrievalFeedbackCollector()

        query_id = collector.start_retrieval(results=["r1", "r2"])
        result = collector.record_click(query_id, "r1")

        assert result is True
        assert "r1" in collector._pending[query_id].clicked

    def test_record_click_unknown_query(self):
        """Click for unknown query returns False."""
        collector = RetrievalFeedbackCollector()

        result = collector.record_click("unknown", "r1")

        assert result is False

    def test_record_dwell(self):
        """Dwell time is recorded correctly."""
        collector = RetrievalFeedbackCollector()

        query_id = collector.start_retrieval(results=["r1"])
        collector.record_dwell(query_id, "r1", 15.0)
        collector.record_dwell(query_id, "r1", 10.0)  # Accumulates

        assert collector._pending[query_id].dwell_times["r1"] == 25.0

    def test_record_dwell_invalid_time(self):
        """Invalid dwell time is rejected."""
        collector = RetrievalFeedbackCollector()

        query_id = collector.start_retrieval(results=["r1"])
        result = collector.record_dwell(query_id, "r1", float("nan"))

        assert result is False

    def test_finalize_retrieval(self):
        """Finalization computes feedback for all results."""
        collector = RetrievalFeedbackCollector()

        query_id = collector.start_retrieval(results=["r1", "r2", "r3"])
        collector.record_click(query_id, "r1")
        collector.record_dwell(query_id, "r1", 30.0)

        feedbacks = collector.finalize_retrieval(query_id)

        assert len(feedbacks) == 3
        assert query_id not in collector._pending

        # Clicked result should have higher relevance
        r1_fb = next(fb for fb in feedbacks if fb.result_id == "r1")
        r3_fb = next(fb for fb in feedbacks if fb.result_id == "r3")
        assert r1_fb.relevance > r3_fb.relevance

    def test_finalize_unknown_query(self):
        """Finalize unknown query returns empty list."""
        collector = RetrievalFeedbackCollector()

        feedbacks = collector.finalize_retrieval("unknown")

        assert feedbacks == []

    def test_record_retrieval_convenience(self):
        """record_retrieval combines all steps."""
        collector = RetrievalFeedbackCollector()

        feedbacks = collector.record_retrieval(
            query_id="q1",
            results=["r1", "r2"],
            clicked=["r1"],
            dwell_times={"r1": 20.0},
        )

        assert len(feedbacks) == 2
        assert feedbacks[0].clicked is True

    def test_compute_relevance_click_only(self):
        """Clicked result gets high relevance."""
        collector = RetrievalFeedbackCollector()

        relevance = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=["r1"],
            dwell_times={},
        )

        assert relevance > 0.5

    def test_compute_relevance_dwell_only(self):
        """Result with dwell time gets moderate relevance."""
        collector = RetrievalFeedbackCollector()

        relevance = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=[],
            dwell_times={"r1": 30.0},
        )

        # Dwell signal (normalized dwell weight ~0.3) gives moderate relevance
        # With position correction, should be above 0.1
        assert relevance > 0.1
        # But without click, should be below click-only
        assert relevance < 0.6

    def test_compute_relevance_no_interaction(self):
        """No interaction gives low relevance."""
        collector = RetrievalFeedbackCollector()

        relevance = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=[],
            dwell_times={},
        )

        assert relevance < 0.5

    def test_compute_relevance_position_bias(self):
        """Position affects relevance computation."""
        collector = RetrievalFeedbackCollector()

        # Same interaction at different positions
        rel_pos0 = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=["r1"],
            dwell_times={"r1": 20.0},
        )

        rel_pos5 = collector.compute_relevance(
            result_id="r2",
            position=5,
            clicked=["r2"],
            dwell_times={"r2": 20.0},
        )

        # Position 5 with same engagement should get boost for position correction
        # (accounting for natural position bias)
        # Both should be high since they got engagement
        assert rel_pos0 > 0.5
        assert rel_pos5 > 0.5

    def test_relevance_bounds(self):
        """Relevance is always in [0, 1]."""
        collector = RetrievalFeedbackCollector()

        # Various edge cases
        test_cases = [
            (0, [], {}),  # No interaction
            (0, ["r1"], {"r1": 1000.0}),  # Extreme dwell
            (10, [], {}),  # Low position
            (0, ["r1"], {}),  # Click only
        ]

        for position, clicked, dwell in test_cases:
            relevance = collector.compute_relevance(
                result_id="r1",
                position=position,
                clicked=clicked,
                dwell_times=dwell,
            )
            assert 0.0 <= relevance <= 1.0

    def test_max_pending_eviction(self):
        """Oldest pending retrieval evicted when limit reached."""
        collector = RetrievalFeedbackCollector(max_pending=2)

        q1 = collector.start_retrieval(results=["r1"])
        q2 = collector.start_retrieval(results=["r2"])
        q3 = collector.start_retrieval(results=["r3"])

        # q1 should have been evicted
        assert q1 not in collector._pending
        assert q3 in collector._pending

    def test_statistics(self):
        """Statistics tracked correctly."""
        collector = RetrievalFeedbackCollector()

        collector.record_retrieval(
            query_id="q1",
            results=["r1", "r2"],
            clicked=["r1"],
            dwell_times={"r1": 10.0},
        )

        stats = collector.get_statistics()

        assert stats["total_retrievals"] == 1
        assert stats["total_feedbacks"] == 2
        assert stats["total_clicks"] == 1

    def test_persistence_sqlite(self):
        """Feedback persisted to SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            collector = RetrievalFeedbackCollector(db_path=db_path)

            collector.record_retrieval(
                query_id="q1",
                results=["r1", "r2"],
                clicked=["r1"],
                dwell_times={"r1": 15.0},
            )

            stats = collector.get_statistics()
            assert stats["persisted_feedbacks"] == 2

            collector.close()

    def test_get_training_batch(self):
        """Training batch retrieved from database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            collector = RetrievalFeedbackCollector(db_path=db_path)

            # Create several feedbacks
            for i in range(5):
                collector.record_retrieval(
                    query_id=f"q{i}",
                    results=[f"r{i}"],
                    clicked=[f"r{i}"],
                    dwell_times={f"r{i}": 30.0},
                )

            batch = collector.get_training_batch(batch_size=3)

            assert len(batch) <= 3
            assert all(isinstance(fb, RetrievalFeedback) for fb in batch)

            collector.close()

    def test_get_feedback_for_result(self):
        """Feedback retrieved for specific result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            collector = RetrievalFeedbackCollector(db_path=db_path)

            # Same result in multiple queries
            for i in range(3):
                collector.record_retrieval(
                    query_id=f"q{i}",
                    results=["shared_result"],
                    clicked=["shared_result"],
                    dwell_times={"shared_result": 10.0 * (i + 1)},
                )

            feedbacks = collector.get_feedback_for_result("shared_result")

            assert len(feedbacks) == 3

            collector.close()

    def test_get_average_relevance(self):
        """Average relevance computed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"
            collector = RetrievalFeedbackCollector(db_path=db_path)

            collector.record_retrieval(
                query_id="q1",
                results=["r1"],
                clicked=["r1"],
                dwell_times={"r1": 30.0},
            )
            collector.record_retrieval(
                query_id="q2",
                results=["r1"],
                clicked=[],
                dwell_times={},
            )

            avg = collector.get_average_relevance("r1")

            assert 0.0 < avg < 1.0

            collector.close()


class TestLearningSignal:
    """Tests for LearningSignal dataclass."""

    def test_creation(self):
        """Learning signal created with required fields."""
        signal = LearningSignal(
            memory_id="mem_123",
            query_id="q_456",
            reward=0.5,
            prediction_error=0.2,
            expected_relevance=0.4,
            actual_relevance=0.6,
            confidence=0.8,
            should_update=True,
        )

        assert signal.memory_id == "mem_123"
        assert signal.reward == 0.5
        assert signal.prediction_error == 0.2

    def test_to_dict(self):
        """Learning signal serializes."""
        signal = LearningSignal(
            memory_id="mem_123",
            query_id="q_456",
            reward=0.5,
            prediction_error=0.2,
            expected_relevance=0.4,
            actual_relevance=0.6,
            confidence=0.8,
            should_update=True,
        )

        d = signal.to_dict()

        assert d["memory_id"] == "mem_123"
        assert d["reward"] == 0.5
        assert "timestamp" in d


class TestFeedbackSignalProcessor:
    """Tests for FeedbackSignalProcessor."""

    def test_initialization(self):
        """Processor initializes with defaults."""
        proc = FeedbackSignalProcessor()

        assert proc.default_expectation == 0.5
        assert proc.expectation_lr == 0.1
        assert len(proc._expectations) == 0

    def test_feedback_to_learning_signal(self):
        """Feedback converts to learning signal."""
        proc = FeedbackSignalProcessor()

        fb = RetrievalFeedback(
            query_id="q1",
            result_id="r1",
            relevance=0.8,
            confidence=0.9,
        )

        signal = proc.feedback_to_learning_signal(fb)

        assert signal.memory_id == "r1"
        assert signal.query_id == "q1"
        assert signal.actual_relevance == 0.8
        # Expected is default (0.5) for first observation
        assert signal.expected_relevance == 0.5
        # PE = 0.8 - 0.5 = 0.3
        assert abs(signal.prediction_error - 0.3) < 0.001

    def test_expectation_updates(self):
        """Expectations update with observations."""
        proc = FeedbackSignalProcessor()

        # First observation
        fb1 = RetrievalFeedback(result_id="r1", relevance=0.8)
        proc.feedback_to_learning_signal(fb1)

        # Expectation should have moved from 0.5 toward 0.8
        exp1 = proc.get_expectation("r1")
        assert exp1 > 0.5
        assert exp1 < 0.8

        # Second observation
        fb2 = RetrievalFeedback(result_id="r1", relevance=0.8)
        proc.feedback_to_learning_signal(fb2)

        exp2 = proc.get_expectation("r1")
        assert exp2 > exp1  # Closer to 0.8

    def test_should_update_thresholds(self):
        """Update decision respects thresholds."""
        proc = FeedbackSignalProcessor(
            min_prediction_error=0.1,
            min_confidence_for_update=0.5,
        )

        # Low PE, high confidence -> no update
        fb1 = RetrievalFeedback(
            result_id="r1",
            relevance=0.51,  # PE = 0.01
            confidence=0.9,
        )
        signal1 = proc.feedback_to_learning_signal(fb1)
        assert signal1.should_update is False

        # High PE, low confidence -> no update
        fb2 = RetrievalFeedback(
            result_id="r2",
            relevance=0.9,  # PE = 0.4
            confidence=0.3,
        )
        signal2 = proc.feedback_to_learning_signal(fb2)
        assert signal2.should_update is False

        # High PE, high confidence -> update
        fb3 = RetrievalFeedback(
            result_id="r3",
            relevance=0.9,
            confidence=0.8,
        )
        signal3 = proc.feedback_to_learning_signal(fb3)
        assert signal3.should_update is True

    def test_reward_transformation(self):
        """Relevance transformed to reward correctly."""
        proc = FeedbackSignalProcessor(reward_scale=2.0)

        # High relevance -> positive reward
        fb1 = RetrievalFeedback(result_id="r1", relevance=0.9, confidence=1.0)
        signal1 = proc.feedback_to_learning_signal(fb1)
        assert signal1.reward > 0

        # Low relevance -> negative reward
        fb2 = RetrievalFeedback(result_id="r2", relevance=0.1, confidence=1.0)
        signal2 = proc.feedback_to_learning_signal(fb2)
        assert signal2.reward < 0

        # Neutral relevance -> near-zero reward
        fb3 = RetrievalFeedback(result_id="r3", relevance=0.5, confidence=1.0)
        signal3 = proc.feedback_to_learning_signal(fb3)
        assert abs(signal3.reward) < 0.1

    def test_process_batch(self):
        """Batch processing works correctly."""
        proc = FeedbackSignalProcessor()

        feedbacks = [
            RetrievalFeedback(result_id=f"r{i}", relevance=i * 0.2)
            for i in range(5)
        ]

        signals = proc.process_batch(feedbacks)

        assert len(signals) == 5
        assert all(isinstance(s, LearningSignal) for s in signals)

    def test_set_expectation(self):
        """Manual expectation setting works."""
        proc = FeedbackSignalProcessor()

        proc.set_expectation("r1", 0.9)
        assert proc.get_expectation("r1") == 0.9

        # Bounds enforced
        proc.set_expectation("r2", 1.5)
        assert proc.get_expectation("r2") == 1.0

    def test_save_load_state(self):
        """State persistence works."""
        proc = FeedbackSignalProcessor()

        # Build some state
        for i in range(3):
            fb = RetrievalFeedback(result_id=f"r{i}", relevance=i * 0.3)
            proc.feedback_to_learning_signal(fb)

        # Save
        state = proc.save_state()

        # Load into new processor
        proc2 = FeedbackSignalProcessor()
        proc2.load_state(state)

        # Verify state matches
        for i in range(3):
            assert proc2.get_expectation(f"r{i}") == proc.get_expectation(f"r{i}")

    def test_statistics(self):
        """Statistics tracked correctly."""
        proc = FeedbackSignalProcessor()

        for i in range(10):
            fb = RetrievalFeedback(
                result_id=f"r{i}",
                relevance=0.9 if i % 2 == 0 else 0.5,
                confidence=0.8,
            )
            proc.feedback_to_learning_signal(fb)

        stats = proc.get_statistics()

        assert stats["total_signals"] == 10
        assert stats["num_tracked_memories"] == 10
        assert stats["update_rate"] > 0

    def test_max_history_eviction(self):
        """Old expectations evicted when max reached."""
        proc = FeedbackSignalProcessor(max_history=3)

        for i in range(5):
            fb = RetrievalFeedback(result_id=f"r{i}", relevance=0.5)
            proc.feedback_to_learning_signal(fb)

        assert len(proc._expectations) <= 3

    def test_clear(self):
        """Clear resets all state."""
        proc = FeedbackSignalProcessor()

        fb = RetrievalFeedback(result_id="r1", relevance=0.8)
        proc.feedback_to_learning_signal(fb)

        proc.clear()

        assert len(proc._expectations) == 0
        assert proc._total_signals == 0


class TestFeedbackToAdapterBridge:
    """Tests for FeedbackToAdapterBridge."""

    def test_initialization(self):
        """Bridge initializes with defaults."""
        bridge = FeedbackToAdapterBridge()

        assert bridge.positive_threshold == 0.6
        assert bridge.negative_threshold == 0.3

    def test_feedback_to_training_signal(self):
        """Feedback converts to training signal."""
        bridge = FeedbackToAdapterBridge()

        feedbacks = [
            RetrievalFeedback(result_id="r1", relevance=0.9, confidence=0.8),  # Positive
            RetrievalFeedback(result_id="r2", relevance=0.5, confidence=0.8),  # Neither
            RetrievalFeedback(result_id="r3", relevance=0.1, confidence=0.8),  # Negative
        ]

        signal = bridge.feedback_to_training_signal(feedbacks)

        assert signal is not None
        assert "r1" in signal.positive_ids
        assert "r3" in signal.negative_ids
        assert "r2" not in signal.positive_ids
        assert "r2" not in signal.negative_ids

    def test_insufficient_data_returns_none(self):
        """Returns None when insufficient positive/negative."""
        bridge = FeedbackToAdapterBridge()

        # Only positives
        feedbacks = [
            RetrievalFeedback(result_id="r1", relevance=0.9, confidence=0.8),
            RetrievalFeedback(result_id="r2", relevance=0.8, confidence=0.8),
        ]

        signal = bridge.feedback_to_training_signal(feedbacks)

        assert signal is None

    def test_confidence_filter(self):
        """Low confidence feedback filtered out."""
        bridge = FeedbackToAdapterBridge(min_confidence=0.7)

        feedbacks = [
            RetrievalFeedback(result_id="r1", relevance=0.9, confidence=0.8),  # Included
            RetrievalFeedback(result_id="r2", relevance=0.9, confidence=0.3),  # Excluded
            RetrievalFeedback(result_id="r3", relevance=0.1, confidence=0.8),  # Included
        ]

        signal = bridge.feedback_to_training_signal(feedbacks)

        assert signal is not None
        assert "r1" in signal.positive_ids
        assert "r2" not in signal.positive_ids

    def test_weights_assigned(self):
        """Weights computed from relevance and confidence."""
        bridge = FeedbackToAdapterBridge()

        feedbacks = [
            RetrievalFeedback(result_id="r1", relevance=0.9, confidence=0.8),
            RetrievalFeedback(result_id="r2", relevance=0.1, confidence=0.8),
        ]

        signal = bridge.feedback_to_training_signal(feedbacks)

        assert signal is not None
        assert signal.positive_weights["r1"] > 0
        assert signal.negative_weights["r2"] > 0


class TestIntegration:
    """Integration tests for the feedback loop system."""

    def test_collector_to_processor_flow(self):
        """Full flow from collection to learning signals."""
        collector = RetrievalFeedbackCollector()
        processor = FeedbackSignalProcessor()

        # Simulate retrieval and interaction
        feedbacks = collector.record_retrieval(
            query_id="q1",
            results=["r1", "r2", "r3"],
            clicked=["r1"],
            dwell_times={"r1": 30.0, "r2": 5.0},
        )

        # Process into learning signals
        signals = processor.process_batch(feedbacks)

        assert len(signals) == 3

        # Clicked result should have positive reward
        r1_signal = next(s for s in signals if s.memory_id == "r1")
        assert r1_signal.reward > 0
        assert r1_signal.should_update is True

        # Non-clicked, low-dwell result should have negative/neutral reward
        r3_signal = next(s for s in signals if s.memory_id == "r3")
        assert r3_signal.reward <= r1_signal.reward

    def test_multiple_queries_learning(self):
        """Learning improves predictions across queries."""
        processor = FeedbackSignalProcessor(expectation_lr=0.3)

        # Memory r1 consistently gets high relevance
        for i in range(5):
            fb = RetrievalFeedback(
                result_id="r1",
                relevance=0.9,
                confidence=0.8,
            )
            signal = processor.feedback_to_learning_signal(fb)

        # Expectation should have moved toward 0.9
        exp = processor.get_expectation("r1")
        assert exp > 0.7

        # Prediction error should be smaller now
        final_fb = RetrievalFeedback(result_id="r1", relevance=0.9, confidence=0.8)
        final_signal = processor.feedback_to_learning_signal(final_fb)
        assert abs(final_signal.prediction_error) < 0.3

    def test_persistence_across_sessions(self):
        """Feedback persists and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "feedback.db"

            # Session 1: Collect feedback with high confidence results
            collector1 = RetrievalFeedbackCollector(db_path=db_path)
            collector1.record_retrieval(
                query_id="q1",
                results=["r1", "r2"],
                clicked=["r1", "r2"],  # Both clicked for high confidence
                dwell_times={"r1": 20.0, "r2": 20.0},
            )
            collector1.close()

            # Session 2: Load and use feedback (lower confidence threshold)
            collector2 = RetrievalFeedbackCollector(db_path=db_path)
            batch = collector2.get_training_batch(batch_size=10, min_confidence=0.3)

            assert len(batch) == 2
            collector2.close()

    def test_dwell_contributes_to_relevance(self):
        """Dwell time contributes positively to relevance score."""
        collector = RetrievalFeedbackCollector()

        # No dwell
        rel_no_dwell = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=[],
            dwell_times={},
        )

        # With dwell
        rel_with_dwell = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=[],
            dwell_times={"r1": 60.0},  # Long dwell
        )

        # Dwell should increase relevance
        assert rel_with_dwell > rel_no_dwell

    def test_click_plus_dwell_highest_relevance(self):
        """Click combined with dwell gives highest relevance."""
        collector = RetrievalFeedbackCollector()

        # Click only
        rel_click = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=["r1"],
            dwell_times={},
        )

        # Click + dwell
        rel_both = collector.compute_relevance(
            result_id="r1",
            position=0,
            clicked=["r1"],
            dwell_times={"r1": 30.0},
        )

        # Both signals should give higher relevance
        assert rel_both >= rel_click
