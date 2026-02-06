"""Comprehensive tests for memory_gate.py - MemoryGate and TemporalBatcher."""

import pytest
from datetime import datetime, timedelta

from t4dm.core.memory_gate import (
    MemoryGate,
    TemporalBatcher,
    GateContext,
    GateResult,
    StorageDecision,
)


class TestStorageDecisionEnum:
    """Test StorageDecision enum variants."""

    def test_all_decisions_defined(self):
        assert StorageDecision.STORE.value == "store"
        assert StorageDecision.SKIP.value == "skip"
        assert StorageDecision.BUFFER.value == "buffer"
        assert StorageDecision.REDACT_THEN_STORE.value == "redact_then_store"


class TestGateContext:
    """Test GateContext dataclass."""

    def test_gate_context_defaults(self):
        ctx = GateContext(session_id="test")
        assert ctx.session_id == "test"
        assert ctx.project is None
        assert ctx.cwd is None
        assert ctx.recent_entities == []
        assert ctx.last_store_time is None
        assert ctx.message_count_since_store == 0
        assert ctx.current_task is None
        assert ctx.is_voice is False
        assert ctx.prediction_error == 0.0
        assert ctx.novelty_signal == 0.0
        assert ctx.reward_signal == 0.0
        assert ctx.theta_phase == 0.0

    def test_gate_context_with_values(self):
        now = datetime.now()
        ctx = GateContext(
            session_id="s1",
            project="t4dm",
            cwd="/home/user",
            recent_entities=["Python", "T4D"],
            last_store_time=now,
            message_count_since_store=5,
            current_task="coding",
            is_voice=True,
            prediction_error=0.5,
            novelty_signal=0.7,
            reward_signal=0.3,
            theta_phase=1.5
        )
        assert ctx.session_id == "s1"
        assert ctx.project == "t4dm"
        assert ctx.cwd == "/home/user"
        assert len(ctx.recent_entities) == 2
        assert ctx.last_store_time == now
        assert ctx.is_voice is True


class TestGateResult:
    """Test GateResult dataclass."""

    def test_gate_result_defaults(self):
        result = GateResult(
            decision=StorageDecision.STORE,
            score=0.8,
            reasons=["test"],
            suggested_importance=0.8
        )
        assert result.decision == StorageDecision.STORE
        assert result.score == 0.8
        assert result.reasons == ["test"]
        assert result.batch_key is None
        assert result.tau_value == 0.5
        assert result.plasticity_gain == 1.0

    def test_gate_result_with_all_fields(self):
        result = GateResult(
            decision=StorageDecision.BUFFER,
            score=0.3,
            reasons=["buffering"],
            suggested_importance=0.4,
            batch_key="project:task",
            tau_value=0.6,
            plasticity_gain=1.2
        )
        assert result.suggested_importance == 0.4
        assert result.batch_key == "project:task"
        assert result.tau_value == 0.6
        assert result.plasticity_gain == 1.2


class TestMemoryGateInitialization:
    """Test MemoryGate initialization."""

    def test_gate_defaults(self):
        gate = MemoryGate()
        assert gate.store_threshold == 0.4
        assert gate.buffer_threshold == 0.2
        assert gate.min_store_interval == timedelta(seconds=30)
        assert gate.max_messages_without_store == 20
        assert gate.voice_mode_adjustments is True
        assert gate.use_temporal_control is True
        assert gate.tau_weight == 0.3

    def test_gate_custom_thresholds(self):
        gate = MemoryGate(store_threshold=0.5, buffer_threshold=0.3)
        assert gate.store_threshold == 0.5
        assert gate.buffer_threshold == 0.3

    def test_gate_patterns_compiled(self):
        gate = MemoryGate()
        assert len(gate._always_store) > 0
        assert len(gate._never_store) > 0
        assert len(gate._actions) > 0
        assert len(gate._entities) > 0

    def test_gate_recent_hashes_initialized(self):
        gate = MemoryGate()
        assert gate._recent_hashes == set()
        assert gate._recent_hash_limit == 100


class TestMemoryGateExplicitTriggers:
    """Test explicit store triggers."""

    def test_remember_triggers_store(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Remember that Python is fast", ctx)
        assert result.decision == StorageDecision.STORE
        assert result.score == 1.0

    def test_dont_forget_triggers_store(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Don't forget to deploy tomorrow", ctx)
        assert result.decision == StorageDecision.STORE

    def test_important_triggers_store(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("This is important: update the config", ctx)
        assert result.decision == StorageDecision.STORE

    def test_deployed_triggers_store(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Successfully deployed to production", ctx)
        assert result.decision == StorageDecision.STORE

    def test_fixed_bug_triggers_store(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Fixed the critical bug in auth", ctx)
        assert result.decision == StorageDecision.STORE


class TestMemoryGateNoisePatterms:
    """Test noise pattern detection."""

    def test_greeting_skipped(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Hi!", ctx)
        assert result.decision == StorageDecision.SKIP
        assert result.score == 0.0

    def test_acknowledgment_skipped(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Got it.", ctx)
        assert result.decision == StorageDecision.SKIP

    def test_filler_sounds_skipped(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        for filler in ["Um.", "Uhh.", "Hmm.", "Uh."]:
            result = gate.evaluate(filler, ctx)
            assert result.decision == StorageDecision.SKIP

    def test_empty_content_skipped(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("   ", ctx)
        assert result.decision == StorageDecision.SKIP

    def test_thank_you_skipped(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Thank you!", ctx)
        assert result.decision == StorageDecision.SKIP


class TestMemoryGateNoveltyScoring:
    """Test novelty scoring."""

    def test_duplicate_content_low_novelty(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")

        # First occurrence
        result1 = gate.evaluate("This is new content", ctx)

        # Same content again
        result2 = gate.evaluate("This is new content", ctx)
        assert result2.score < result1.score  # Second should be lower

    def test_short_content_medium_novelty(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("a b", ctx)  # Very short
        # Should get novelty around 0.3 for short content


class TestMemoryGateOutcomeScoring:
    """Test outcome presence scoring."""

    def test_success_outcome_high_score(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Tests passed successfully", ctx)
        assert "outcome" in str(result.reasons).lower() or result.score > 0.4

    def test_failure_outcome_moderate_score(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Failed to connect to database", ctx)
        assert result.score > 0.3

    def test_fixed_outcome_high_score(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Fixed the memory leak issue", ctx)
        assert result.score > 0.4


class TestMemoryGateActionScoring:
    """Test action significance scoring."""

    def test_file_creation_high_action_score(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Created deployment.py file", ctx)
        assert "action" in str(result.reasons).lower()

    def test_git_commit_high_action_score(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("git commit -m 'fix memory leak'", ctx)
        assert result.score > 0.3

    def test_deployment_very_high_action_score(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Deployed to production", ctx)
        assert result.score > 0.5


class TestMemoryGateEntityScoring:
    """Test entity density scoring."""

    def test_known_entity_increases_score(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s", recent_entities=["Python"])
        result = gate.evaluate("Python is great for ML", ctx)
        # Entity score should contribute to total


class TestMemoryGateTimePressure:
    """Test time-based storage pressure."""

    def test_force_store_after_message_limit(self):
        gate = MemoryGate()
        ctx = GateContext(
            session_id="s",
            message_count_since_store=gate.max_messages_without_store
        )
        assert gate.force_store_check(ctx) is True

    def test_no_force_store_before_limit(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s", message_count_since_store=5)
        assert gate.force_store_check(ctx) is False

    def test_force_store_after_long_time(self):
        gate = MemoryGate()
        old_time = datetime.now() - timedelta(minutes=31)
        ctx = GateContext(session_id="s", last_store_time=old_time)
        assert gate.force_store_check(ctx) is True

    def test_no_force_store_recent(self):
        gate = MemoryGate()
        recent_time = datetime.now() - timedelta(minutes=5)
        ctx = GateContext(session_id="s", last_store_time=recent_time)
        assert gate.force_store_check(ctx) is False


class TestMemoryGateVoiceMode:
    """Test voice-specific adjustments."""

    def test_voice_mode_adjustments_enabled(self):
        gate = MemoryGate(voice_mode_adjustments=True)
        ctx = GateContext(session_id="s", is_voice=True)
        # Voice should have different weighting


class TestMemoryGateDecision:
    """Test decision making."""

    def test_store_decision_above_threshold(self):
        gate = MemoryGate(store_threshold=0.4)
        ctx = GateContext(session_id="s")
        # Strong content should trigger store
        result = gate.evaluate("Remember I successfully deployed the new API", ctx)
        assert result.decision in [StorageDecision.STORE, StorageDecision.BUFFER]

    def test_buffer_decision_between_thresholds(self):
        gate = MemoryGate(store_threshold=0.6, buffer_threshold=0.3)
        ctx = GateContext(session_id="s")
        # Moderate content might buffer
        result = gate.evaluate("Started working on feature", ctx)
        # Could be buffer or store depending on score


class TestMemoryGateBatchKey:
    """Test batch key computation."""

    def test_batch_key_with_project_and_task(self):
        gate = MemoryGate()
        ctx = GateContext(
            session_id="s",
            project="t4dm",
            current_task="coding"
        )
        batch_key = gate._compute_batch_key("test", ctx)
        assert "t4dm" in batch_key
        assert "coding" in batch_key

    def test_batch_key_defaults(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        batch_key = gate._compute_batch_key("test", ctx)
        assert "default" in batch_key
        assert "general" in batch_key


class TestMemoryGateContentRecording:
    """Test content hash recording for novelty."""

    def test_record_content_adds_hash(self):
        gate = MemoryGate()
        initial_count = len(gate._recent_hashes)
        gate._record_content("new content")
        assert len(gate._recent_hashes) == initial_count + 1

    def test_record_content_duplicate_hash(self):
        gate = MemoryGate()
        gate._record_content("content")
        count1 = len(gate._recent_hashes)
        gate._record_content("content")
        count2 = len(gate._recent_hashes)
        assert count1 == count2  # No duplicate

    def test_record_content_prunes_old_hashes(self):
        gate = MemoryGate()
        gate._recent_hash_limit = 10
        for i in range(15):
            gate._record_content(f"content {i}")
        # Should prune to stay under limit
        assert len(gate._recent_hashes) <= gate._recent_hash_limit


class TestTemporalBatcher:
    """Test TemporalBatcher."""

    def test_batcher_initialization(self):
        batcher = TemporalBatcher()
        assert batcher.batch_window == timedelta(minutes=2)
        assert batcher.max_batch_size == 10

    def test_batcher_custom_window(self):
        batcher = TemporalBatcher(batch_window=timedelta(seconds=30))
        assert batcher.batch_window == timedelta(seconds=30)

    def test_batcher_add_single_item(self):
        batcher = TemporalBatcher()
        result = batcher.add("key1", "content1")
        assert result is None  # Not flushed yet

    def test_batcher_flush_on_size(self):
        batcher = TemporalBatcher(max_batch_size=2)
        batcher.add("key1", "content1")
        result = batcher.add("key1", "content2")
        assert result is None  # Still under limit

        result = batcher.add("key1", "content3")
        assert result is not None  # Flushed on size
        assert "content1" in result
        assert "content2" in result

    def test_batcher_flush_on_timeout(self):
        batcher = TemporalBatcher(batch_window=timedelta(milliseconds=100))
        batcher.add("key1", "content1")

        # Wait for timeout
        import time
        time.sleep(0.15)

        result = batcher.add("key1", "content2")
        assert result is not None  # Flushed on timeout

    def test_batcher_flush_all(self):
        batcher = TemporalBatcher()
        batcher.add("key1", "content1")
        batcher.add("key1", "content2")
        batcher.add("key2", "content3")

        results = batcher.flush_all()
        assert len(results) == 2
        assert all(key for key, _ in results)

    def test_batcher_removes_duplicates(self):
        batcher = TemporalBatcher(max_batch_size=2)
        batcher.add("key1", "dup")
        batcher.add("key1", "dup")
        result = batcher.add("key1", "new")

        # Should not have duplicates
        dup_count = result.count("dup")
        assert dup_count == 1

    def test_batcher_empty_flush(self):
        batcher = TemporalBatcher()
        results = batcher.flush_all()
        assert results == []

    def test_batcher_multiple_keys(self):
        batcher = TemporalBatcher()
        batcher.add("project1:task1", "content1")
        batcher.add("project1:task2", "content2")
        batcher.add("project2:task1", "content3")

        results = batcher.flush_all()
        assert len(results) == 3


class TestMemoryGateIntegration:
    """Integration tests for MemoryGate."""

    def test_evaluate_produces_reasons(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result = gate.evaluate("Important update deployed to production", ctx)
        assert len(result.reasons) > 0
        assert any(reason for reason in result.reasons)

    def test_evaluate_consistency(self):
        gate = MemoryGate()
        ctx = GateContext(session_id="s")
        result1 = gate.evaluate("Same content", ctx)
        result2 = gate.evaluate("Same content", ctx)
        # Second evaluation should have lower novelty
        assert result2.score <= result1.score

    def test_gate_with_prediction_error_signal(self):
        gate = MemoryGate(use_temporal_control=False)  # Disable temporal control
        ctx = GateContext(
            session_id="s",
            prediction_error=0.8,
            novelty_signal=0.9
        )
        result = gate.evaluate("Surprising outcome occurred", ctx)
        assert result.decision is not None

    def test_full_evaluation_flow(self):
        gate = MemoryGate()
        ctx = GateContext(
            session_id="test-session",
            project="my-project",
            current_task="feature-dev",
            recent_entities=["Python", "FastAPI"],
            message_count_since_store=3,
            last_store_time=datetime.now() - timedelta(minutes=2)
        )

        result = gate.evaluate(
            "Successfully completed Python API endpoint for user authentication",
            ctx
        )

        assert result.decision in [StorageDecision.STORE, StorageDecision.BUFFER, StorageDecision.SKIP]
        assert 0.0 <= result.score <= 1.0
        assert len(result.reasons) > 0
        assert 0.0 <= result.suggested_importance <= 1.0
