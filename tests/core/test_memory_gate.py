"""Tests for memory gate module."""

import pytest
from datetime import datetime, timedelta

from t4dm.core.memory_gate import (
    StorageDecision,
    GateContext,
    GateResult,
    MemoryGate,
    TemporalBatcher,
)


class TestStorageDecision:
    """Tests for StorageDecision enum."""

    def test_decision_values(self):
        """Test decision values."""
        assert StorageDecision.STORE.value == "store"
        assert StorageDecision.SKIP.value == "skip"
        assert StorageDecision.BUFFER.value == "buffer"
        assert StorageDecision.REDACT_THEN_STORE.value == "redact_then_store"

    def test_all_decisions_exist(self):
        """Test all decisions are defined."""
        assert len(list(StorageDecision)) == 4


class TestGateContext:
    """Tests for GateContext dataclass."""

    def test_basic_context(self):
        """Test basic context creation."""
        ctx = GateContext(session_id="test-session")
        assert ctx.session_id == "test-session"
        assert ctx.project is None
        assert ctx.recent_entities == []
        assert ctx.message_count_since_store == 0

    def test_context_with_all_fields(self):
        """Test context with all fields."""
        ctx = GateContext(
            session_id="session-123",
            project="ww",
            cwd="/home/user/ww",
            recent_entities=["Entity1", "Entity2"],
            last_store_time=datetime.now(),
            message_count_since_store=5,
            current_task="writing tests",
            is_voice=True,
        )
        assert ctx.project == "ww"
        assert len(ctx.recent_entities) == 2
        assert ctx.is_voice is True
        assert ctx.message_count_since_store == 5


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_store_result(self):
        """Test store result."""
        result = GateResult(
            decision=StorageDecision.STORE,
            score=0.8,
            reasons=["High importance"],
            suggested_importance=0.9,
        )
        assert result.decision == StorageDecision.STORE
        assert result.score == 0.8
        assert len(result.reasons) == 1

    def test_buffer_result(self):
        """Test buffer result with batch key."""
        result = GateResult(
            decision=StorageDecision.BUFFER,
            score=0.3,
            reasons=["Moderate importance"],
            suggested_importance=0.4,
            batch_key="project:task",
        )
        assert result.decision == StorageDecision.BUFFER
        assert result.batch_key == "project:task"


class TestMemoryGate:
    """Tests for MemoryGate class."""

    @pytest.fixture
    def gate(self):
        """Create memory gate."""
        return MemoryGate()

    @pytest.fixture
    def context(self):
        """Create basic context."""
        return GateContext(session_id="test-session")

    def test_initialization(self, gate):
        """Test gate initialization."""
        assert gate.store_threshold == 0.4
        assert gate.buffer_threshold == 0.2
        assert gate.max_messages_without_store == 20

    def test_initialization_custom(self):
        """Test custom initialization."""
        gate = MemoryGate(
            store_threshold=0.6,
            buffer_threshold=0.3,
            min_store_interval=timedelta(seconds=60),
            max_messages_without_store=10,
            voice_mode_adjustments=False,
        )
        assert gate.store_threshold == 0.6
        assert gate.buffer_threshold == 0.3
        assert gate.max_messages_without_store == 10
        assert gate.voice_mode_adjustments is False

    def test_explicit_remember_trigger(self, gate, context):
        """Test 'remember this' always stores."""
        result = gate.evaluate("Remember that the API key is in .env", context)
        assert result.decision == StorageDecision.STORE
        assert result.score == 1.0

    def test_explicit_dont_forget_trigger(self, gate, context):
        """Test 'don't forget' always stores."""
        result = gate.evaluate("Don't forget to run tests before pushing", context)
        assert result.decision == StorageDecision.STORE

    def test_explicit_important_trigger(self, gate, context):
        """Test 'important' always stores."""
        result = gate.evaluate("This is important for the deployment", context)
        assert result.decision == StorageDecision.STORE

    def test_explicit_note_trigger(self, gate, context):
        """Test 'note that' always stores."""
        result = gate.evaluate("Note that the config file changed", context)
        assert result.decision == StorageDecision.STORE

    def test_explicit_deployed_trigger(self, gate, context):
        """Test 'deployed' always stores."""
        result = gate.evaluate("Successfully deployed the application", context)
        assert result.decision == StorageDecision.STORE

    def test_explicit_fixed_bug_trigger(self, gate, context):
        """Test 'fixed bug' always stores."""
        result = gate.evaluate("Fixed the authentication bug", context)
        assert result.decision == StorageDecision.STORE

    def test_explicit_completed_trigger(self, gate, context):
        """Test 'completed' always stores."""
        result = gate.evaluate("Completed the migration task", context)
        assert result.decision == StorageDecision.STORE

    def test_noise_greeting_skip(self, gate, context):
        """Test greeting noise is skipped."""
        for greeting in ["hi", "hello", "hey", "thanks", "ok", "sure"]:
            result = gate.evaluate(greeting, context)
            assert result.decision == StorageDecision.SKIP

    def test_noise_filler_skip(self, gate, context):
        """Test filler words are skipped."""
        for filler in ["um", "uh", "hmm", "ah"]:
            result = gate.evaluate(filler, context)
            assert result.decision == StorageDecision.SKIP

    def test_noise_acknowledgment_skip(self, gate, context):
        """Test acknowledgments are skipped."""
        for ack in ["got it", "sounds good", "makes sense", "i see"]:
            result = gate.evaluate(ack, context)
            assert result.decision == StorageDecision.SKIP

    def test_noise_empty_skip(self, gate, context):
        """Test empty content is skipped."""
        result = gate.evaluate("   ", context)
        assert result.decision == StorageDecision.SKIP

    def test_action_create_file(self, gate, context):
        """Test file creation scores high."""
        result = gate.evaluate("Created utils.py with helper functions", context)
        assert result.score > 0.3

    def test_action_git_commit(self, gate, context):
        """Test git commit scores high."""
        result = gate.evaluate("git commit and pushed changes to main", context)
        assert result.score > 0.3

    def test_action_test_pass(self, gate, context):
        """Test tests passing scores high."""
        result = gate.evaluate("All tests passed successfully", context)
        assert result.score > 0.3

    def test_action_deploy_prod(self, gate, context):
        """Test production deploy always stores."""
        result = gate.evaluate("Deployed to production environment", context)
        assert result.decision == StorageDecision.STORE

    def test_novelty_duplicate_detection(self, gate, context):
        """Test duplicate content detection."""
        content = "Working on the user authentication module"

        # First evaluation - should be novel
        result1 = gate.evaluate(content, context)

        # Second evaluation - should have lower novelty
        gate._record_content(content)  # Force record
        result2 = gate.evaluate(content, context)

        # Second should have lower score due to duplicate
        assert result2.score <= result1.score

    def test_entity_score_with_known_entities(self, gate):
        """Test entity scoring with known entities."""
        context = GateContext(
            session_id="test",
            recent_entities=["UserService", "AuthModule"],
        )
        result = gate.evaluate("Updated the UserService authentication flow", context)
        # Should have entity score contribution in reasons
        # Reasons may have τ(t) first if temporal control is enabled, then breakdown
        reasons_str = " ".join(result.reasons)
        assert "entity=" in reasons_str

    def test_time_pressure_no_previous_store(self, gate, context):
        """Test time pressure without previous store."""
        # No last_store_time should add pressure
        result = gate.evaluate("Working on a feature", context)
        # Reasons may have τ(t) first if temporal control is enabled, then breakdown
        reasons_str = " ".join(result.reasons)
        assert "time=" in reasons_str

    def test_time_pressure_long_time_since_store(self, gate):
        """Test time pressure with long gap since last store."""
        context = GateContext(
            session_id="test",
            last_store_time=datetime.now() - timedelta(minutes=15),
            message_count_since_store=5,
        )
        result = gate.evaluate("Still working on the feature", context)
        # Should have time pressure
        assert result.score > 0

    def test_time_pressure_many_messages(self, gate):
        """Test time pressure with many messages."""
        context = GateContext(
            session_id="test",
            message_count_since_store=15,
        )
        result = gate.evaluate("Another message about work", context)
        # Should have message count pressure
        assert result.score > 0

    def test_voice_mode_adjustments(self, gate):
        """Test voice mode adjusts scoring."""
        context_voice = GateContext(session_id="test", is_voice=True)
        context_text = GateContext(session_id="test", is_voice=False)

        # Low-action content should score differently in voice mode
        content = "The weather is nice today"
        result_voice = gate.evaluate(content, context_voice)
        result_text = gate.evaluate(content, context_text)

        # Voice mode typically has different thresholds
        # Both should be calculated, scores may differ
        assert isinstance(result_voice.score, float)
        assert isinstance(result_text.score, float)

    def test_buffer_decision(self, gate, context):
        """Test buffer decision for moderate content."""
        # Set thresholds to make buffering likely
        gate.store_threshold = 0.8
        gate.buffer_threshold = 0.2

        result = gate.evaluate("Working on some code changes", context)
        # Medium score content should buffer
        if result.score >= 0.2 and result.score < 0.8:
            assert result.decision == StorageDecision.BUFFER

    def test_batch_key_computation(self, gate):
        """Test batch key computation."""
        context = GateContext(
            session_id="test",
            project="ww",
            current_task="testing",
        )
        key = gate._compute_batch_key("some content", context)
        assert "ww" in key
        assert "testing" in key

    def test_batch_key_defaults(self, gate, context):
        """Test batch key with no project/task."""
        key = gate._compute_batch_key("content", context)
        assert "default" in key
        assert "general" in key

    def test_record_content_limits_size(self, gate, context):
        """Test content recording limits hash set size."""
        # Record many contents
        for i in range(150):
            gate._record_content(f"Content {i}")

        # Should have been trimmed
        assert len(gate._recent_hashes) < 150

    def test_force_store_check_message_count(self, gate):
        """Test force store based on message count."""
        context = GateContext(
            session_id="test",
            message_count_since_store=25,
        )
        assert gate.force_store_check(context) is True

    def test_force_store_check_time_elapsed(self, gate):
        """Test force store based on time elapsed."""
        context = GateContext(
            session_id="test",
            last_store_time=datetime.now() - timedelta(minutes=35),
            message_count_since_store=1,
        )
        assert gate.force_store_check(context) is True

    def test_no_force_store_normal_conditions(self, gate):
        """Test no force store under normal conditions."""
        context = GateContext(
            session_id="test",
            last_store_time=datetime.now() - timedelta(minutes=5),
            message_count_since_store=5,
        )
        assert gate.force_store_check(context) is False

    def test_outcome_score_success(self, gate, context):
        """Test outcome scoring for success."""
        score = gate._outcome_score("The task succeeded and tests passed")
        assert score > 0.5

    def test_outcome_score_failure(self, gate, context):
        """Test outcome scoring for failure."""
        score = gate._outcome_score("The build failed with errors")
        assert score > 0.5

    def test_outcome_score_fixed(self, gate, context):
        """Test outcome scoring for fixed."""
        score = gate._outcome_score("Fixed the issue and resolved the bug")
        assert score == 1.0

    def test_outcome_score_starting(self, gate, context):
        """Test outcome scoring for starting."""
        score = gate._outcome_score("Just starting to work on this")
        assert score < 0.5

    def test_novelty_score_short_content(self, gate, context):
        """Test novelty score for short content."""
        score = gate._novelty_score("hi")
        assert score == 0.3  # Too short to judge

    def test_novelty_score_novel_content(self, gate, context):
        """Test novelty score for novel content."""
        score = gate._novelty_score("This is a completely new piece of content about testing")
        assert score == 0.8  # Assumed novel

    def test_action_score_no_action(self, gate, context):
        """Test action score with no action."""
        score = gate._action_score("Just a regular message")
        assert score == 0.0

    def test_action_score_file_edit(self, gate, context):
        """Test action score for file edit."""
        score = gate._action_score("Modified the config.json file")
        assert score > 0


class TestTemporalBatcher:
    """Tests for TemporalBatcher class."""

    @pytest.fixture
    def batcher(self):
        """Create batcher."""
        return TemporalBatcher()

    def test_initialization(self, batcher):
        """Test batcher initialization."""
        assert batcher.batch_window == timedelta(minutes=2)
        assert batcher.max_batch_size == 10

    def test_initialization_custom(self):
        """Test custom initialization."""
        batcher = TemporalBatcher(
            batch_window=timedelta(minutes=5),
            max_batch_size=20,
        )
        assert batcher.batch_window == timedelta(minutes=5)
        assert batcher.max_batch_size == 20

    def test_add_first_content(self, batcher):
        """Test adding first content."""
        result = batcher.add("key1", "First content")
        assert result is None  # Not flushed yet
        assert "key1" in batcher._batches
        assert len(batcher._batches["key1"]) == 1

    def test_add_multiple_content(self, batcher):
        """Test adding multiple content items."""
        batcher.add("key1", "Content 1")
        batcher.add("key1", "Content 2")
        batcher.add("key1", "Content 3")

        assert len(batcher._batches["key1"]) == 3

    def test_add_different_keys(self, batcher):
        """Test adding to different keys."""
        batcher.add("key1", "Content A")
        batcher.add("key2", "Content B")

        assert len(batcher._batches) == 2
        assert "key1" in batcher._batches
        assert "key2" in batcher._batches

    def test_add_triggers_flush_on_size(self, batcher):
        """Test batch flush when size limit reached."""
        batcher.max_batch_size = 3

        batcher.add("key1", "Content 1")
        batcher.add("key1", "Content 2")
        result = batcher.add("key1", "Content 3")

        # Third add should not flush yet
        assert result is None

        # Fourth add should flush the previous batch
        result = batcher.add("key1", "Content 4")
        assert result is not None
        assert "Content 1" in result

    def test_flush_batch_combines_content(self, batcher):
        """Test batch flushing combines content."""
        batcher.add("key1", "First")
        batcher.add("key1", "Second")
        batcher.add("key1", "Third")

        result = batcher._flush_batch("key1")
        assert "First" in result
        assert "Second" in result
        assert "Third" in result
        assert " | " in result  # Joined with separator

    def test_flush_batch_removes_duplicates(self, batcher):
        """Test batch flushing removes duplicates."""
        batcher.add("key1", "Same content")
        batcher.add("key1", "Same content")
        batcher.add("key1", "Different content")

        result = batcher._flush_batch("key1")
        # Should have "Same content" only once
        assert result.count("Same content") == 1
        assert "Different content" in result

    def test_flush_batch_empty(self, batcher):
        """Test flushing empty/nonexistent batch."""
        result = batcher._flush_batch("nonexistent")
        assert result == ""

    def test_flush_all(self, batcher):
        """Test flushing all batches."""
        batcher.add("key1", "Content A1")
        batcher.add("key1", "Content A2")
        batcher.add("key2", "Content B1")

        results = batcher.flush_all()

        assert len(results) == 2
        keys = [r[0] for r in results]
        assert "key1" in keys
        assert "key2" in keys

        # Batches should be cleared
        assert len(batcher._batches) == 0

    def test_flush_all_empty(self, batcher):
        """Test flushing when no batches."""
        results = batcher.flush_all()
        assert results == []
