"""
Tests for Session Lifecycle Hooks (Phase 10).

Tests the Claude Agent SDK integration hooks for session management.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from t4dm.hooks.base import HookContext
from t4dm.hooks.session_lifecycle import (
    IdleConsolidationHook,
    SessionContext,
    SessionEndHook,
    SessionStartHook,
    TaskOutcomeHook,
    create_session_hooks,
)
from t4dm.sdk.agent_client import CreditAssignmentResult, ScoredMemory
from t4dm.sdk.models import Episode, EpisodeContext


@pytest.fixture
def mock_episode():
    """Create a mock episode."""
    return Episode(
        id=uuid4(),
        session_id="test-session",
        content="Test memory content",
        timestamp=datetime.now(),
        outcome="success",
        emotional_valence=0.7,
        context=EpisodeContext(),
        access_count=1,
        stability=0.5,
    )


@pytest.fixture
def mock_memory_client():
    """Create a mock memory client."""
    client = AsyncMock()
    client.session_id = "test-session"
    client.get_stats.return_value = {"session_id": "test-session"}
    return client


@pytest.fixture
def hook_context():
    """Create a basic hook context."""
    return HookContext(
        session_id="test-session",
        metadata={"project": "test-project"},
    )


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_creation(self):
        """Test session context creation."""
        ctx = SessionContext(
            session_id="test-123",
            project="my-project",
        )
        assert ctx.session_id == "test-123"
        assert ctx.project == "my-project"
        assert ctx.task_count == 0
        assert ctx.successful_tasks == 0
        assert ctx.start_time is not None

    def test_default_values(self):
        """Test default values."""
        ctx = SessionContext(session_id="test")
        assert ctx.end_time is None
        assert ctx.memories_retrieved == 0
        assert ctx.memories_stored == 0
        assert ctx.consolidations == 0
        assert ctx.metadata == {}


class TestSessionStartHook:
    """Tests for SessionStartHook."""

    def test_init_defaults(self):
        """Test default initialization."""
        hook = SessionStartHook()
        assert hook.name == "session_start"
        assert hook._load_context is True
        assert hook._max_context_items == 5

    def test_init_custom(self, mock_memory_client):
        """Test custom initialization."""
        hook = SessionStartHook(
            memory_client=mock_memory_client,
            load_context=False,
            max_context_items=10,
        )
        assert hook._memory == mock_memory_client
        assert hook._load_context is False
        assert hook._max_context_items == 10

    @pytest.mark.asyncio
    async def test_execute_creates_session_context(self, hook_context):
        """Test execution creates session context."""
        hook = SessionStartHook(load_context=False)

        result = await hook.execute(hook_context)

        assert result.session_id is not None
        assert "session_context" in result.metadata
        session_ctx = result.metadata["session_context"]
        assert isinstance(session_ctx, SessionContext)

    @pytest.mark.asyncio
    async def test_execute_loads_context(self, mock_memory_client, mock_episode, hook_context):
        """Test execution loads memory context."""
        mock_memory_client.retrieve_for_task.return_value = [
            ScoredMemory(episode=mock_episode, similarity_score=0.9)
        ]

        hook = SessionStartHook(
            memory_client=mock_memory_client,
            load_context=True,
        )

        result = await hook.execute(hook_context)

        mock_memory_client.retrieve_for_task.assert_called_once()
        assert "loaded_memories" in result.output_data
        assert len(result.output_data["loaded_memories"]) == 1

    @pytest.mark.asyncio
    async def test_execute_custom_context_query(self, mock_memory_client, hook_context):
        """Test custom context query."""
        mock_memory_client.retrieve_for_task.return_value = []

        hook = SessionStartHook(
            memory_client=mock_memory_client,
            context_query="custom query",
        )

        await hook.execute(hook_context)

        call_args = mock_memory_client.retrieve_for_task.call_args
        assert call_args.kwargs["query"] == "custom query"

    @pytest.mark.asyncio
    async def test_execute_callback(self, mock_memory_client, mock_episode, hook_context):
        """Test on_context_loaded callback."""
        mock_memory_client.retrieve_for_task.return_value = [
            ScoredMemory(episode=mock_episode, similarity_score=0.9)
        ]

        callback_called = False
        callback_memories = None

        async def on_loaded(memories):
            nonlocal callback_called, callback_memories
            callback_called = True
            callback_memories = memories

        hook = SessionStartHook(
            memory_client=mock_memory_client,
            on_context_loaded=on_loaded,
        )

        await hook.execute(hook_context)

        assert callback_called
        assert len(callback_memories) == 1

    def test_get_session(self):
        """Test getting session by ID."""
        hook = SessionStartHook()
        hook._sessions["test-123"] = SessionContext(session_id="test-123")

        session = hook.get_session("test-123")
        assert session is not None
        assert session.session_id == "test-123"

        assert hook.get_session("unknown") is None


class TestSessionEndHook:
    """Tests for SessionEndHook."""

    def test_init_defaults(self):
        """Test default initialization."""
        hook = SessionEndHook()
        assert hook.name == "session_end"
        assert hook._auto_consolidate is True
        assert hook._consolidation_mode == "deep"
        assert hook._store_summary is True

    def test_init_custom(self, mock_memory_client):
        """Test custom initialization."""
        hook = SessionEndHook(
            memory_client=mock_memory_client,
            auto_consolidate=False,
            consolidation_mode="light",
            store_summary=False,
        )
        assert hook._auto_consolidate is False
        assert hook._consolidation_mode == "light"
        assert hook._store_summary is False

    @pytest.mark.asyncio
    async def test_execute_without_session_context(self, hook_context):
        """Test execution without session context."""
        hook = SessionEndHook()
        hook_context.metadata = {}

        result = await hook.execute(hook_context)

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_sets_end_time(self, mock_memory_client, hook_context):
        """Test execution sets end time."""
        session_ctx = SessionContext(session_id="test")
        hook_context.metadata["session_context"] = session_ctx

        mock_memory_client.store_experience.return_value = MagicMock()
        mock_memory_client.trigger_consolidation.return_value = {}

        hook = SessionEndHook(memory_client=mock_memory_client)

        await hook.execute(hook_context)

        assert session_ctx.end_time is not None

    @pytest.mark.asyncio
    async def test_execute_stores_summary(self, mock_memory_client, hook_context):
        """Test execution stores session summary."""
        session_ctx = SessionContext(session_id="test", task_count=5, successful_tasks=4)
        hook_context.metadata["session_context"] = session_ctx

        mock_memory_client.store_experience.return_value = MagicMock()
        mock_memory_client.trigger_consolidation.return_value = {}

        hook = SessionEndHook(
            memory_client=mock_memory_client,
            store_summary=True,
        )

        await hook.execute(hook_context)

        mock_memory_client.store_experience.assert_called_once()
        call_args = mock_memory_client.store_experience.call_args
        assert "5 tasks" in call_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_execute_triggers_consolidation(self, mock_memory_client, hook_context):
        """Test execution triggers consolidation."""
        session_ctx = SessionContext(session_id="test")
        hook_context.metadata["session_context"] = session_ctx

        mock_memory_client.store_experience.return_value = MagicMock()
        mock_memory_client.trigger_consolidation.return_value = {}

        hook = SessionEndHook(
            memory_client=mock_memory_client,
            auto_consolidate=True,
            consolidation_mode="full",
        )

        await hook.execute(hook_context)

        mock_memory_client.trigger_consolidation.assert_called_with(mode="full")

    @pytest.mark.asyncio
    async def test_execute_output_stats(self, mock_memory_client, hook_context):
        """Test execution outputs session stats."""
        session_ctx = SessionContext(
            session_id="test",
            task_count=10,
            successful_tasks=8,
            memories_retrieved=20,
        )
        hook_context.metadata["session_context"] = session_ctx

        mock_memory_client.store_experience.return_value = MagicMock()
        mock_memory_client.trigger_consolidation.return_value = {}

        hook = SessionEndHook(memory_client=mock_memory_client)

        result = await hook.execute(hook_context)

        assert "session_stats" in result.output_data
        stats = result.output_data["session_stats"]
        assert stats["task_count"] == 10
        assert stats["successful_tasks"] == 8

    @pytest.mark.asyncio
    async def test_execute_callback(self, mock_memory_client, hook_context):
        """Test on_session_end callback."""
        session_ctx = SessionContext(session_id="test")
        hook_context.metadata["session_context"] = session_ctx

        mock_memory_client.store_experience.return_value = MagicMock()
        mock_memory_client.trigger_consolidation.return_value = {}

        callback_ctx = None

        async def on_end(ctx):
            nonlocal callback_ctx
            callback_ctx = ctx

        hook = SessionEndHook(
            memory_client=mock_memory_client,
            on_session_end=on_end,
        )

        await hook.execute(hook_context)

        assert callback_ctx is not None
        assert callback_ctx.session_id == "test"


class TestTaskOutcomeHook:
    """Tests for TaskOutcomeHook."""

    def test_init_defaults(self):
        """Test default initialization."""
        hook = TaskOutcomeHook()
        assert hook.name == "task_outcome"
        assert hook._auto_store_experience is True

    @pytest.mark.asyncio
    async def test_execute_without_task_id(self, hook_context):
        """Test execution without task_id."""
        hook = TaskOutcomeHook()
        hook_context.metadata = {}

        result = await hook.execute(hook_context)

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_reports_outcome(self, mock_memory_client, hook_context):
        """Test execution reports task outcome."""
        hook_context.metadata = {
            "task_id": "task-123",
            "success": True,
            "session_context": SessionContext(session_id="test"),
        }

        mock_memory_client.report_task_outcome.return_value = CreditAssignmentResult(
            credited=3,
            reconsolidated=["mem-1"],
            total_lr_applied=0.02,
        )

        hook = TaskOutcomeHook(memory_client=mock_memory_client)

        result = await hook.execute(hook_context)

        mock_memory_client.report_task_outcome.assert_called_once()
        call_args = mock_memory_client.report_task_outcome.call_args
        assert call_args.kwargs["task_id"] == "task-123"
        assert call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_execute_partial_credit(self, mock_memory_client, hook_context):
        """Test execution with partial credit."""
        hook_context.metadata = {
            "task_id": "task-456",
            "partial_credit": 0.7,
            "session_context": SessionContext(session_id="test"),
        }

        mock_memory_client.report_task_outcome.return_value = CreditAssignmentResult(
            credited=2,
        )

        hook = TaskOutcomeHook(memory_client=mock_memory_client)

        await hook.execute(hook_context)

        call_args = mock_memory_client.report_task_outcome.call_args
        assert call_args.kwargs["partial_credit"] == 0.7

    @pytest.mark.asyncio
    async def test_execute_stores_experience(self, mock_memory_client, mock_episode, hook_context):
        """Test execution stores task experience."""
        hook_context.metadata = {
            "task_id": "task-789",
            "success": True,
            "task_content": "This was the task content",
            "session_context": SessionContext(session_id="test"),
        }

        mock_memory_client.report_task_outcome.return_value = CreditAssignmentResult(credited=1)
        mock_memory_client.store_experience.return_value = mock_episode

        hook = TaskOutcomeHook(
            memory_client=mock_memory_client,
            auto_store_experience=True,
        )

        await hook.execute(hook_context)

        mock_memory_client.store_experience.assert_called_once()
        call_args = mock_memory_client.store_experience.call_args
        assert call_args.kwargs["content"] == "This was the task content"
        assert call_args.kwargs["outcome"] == "success"

    @pytest.mark.asyncio
    async def test_execute_updates_session_context(self, mock_memory_client, hook_context):
        """Test execution updates session context."""
        session_ctx = SessionContext(session_id="test")
        hook_context.metadata = {
            "task_id": "task-123",
            "success": True,
            "session_context": session_ctx,
        }

        mock_memory_client.report_task_outcome.return_value = CreditAssignmentResult(credited=1)

        hook = TaskOutcomeHook(memory_client=mock_memory_client)

        await hook.execute(hook_context)

        assert session_ctx.task_count == 1
        assert session_ctx.successful_tasks == 1

    @pytest.mark.asyncio
    async def test_execute_failure_outcome(self, mock_memory_client, hook_context):
        """Test execution with failure outcome."""
        session_ctx = SessionContext(session_id="test")
        hook_context.metadata = {
            "task_id": "task-fail",
            "success": False,
            "session_context": session_ctx,
        }

        mock_memory_client.report_task_outcome.return_value = CreditAssignmentResult(credited=1)

        hook = TaskOutcomeHook(memory_client=mock_memory_client)

        await hook.execute(hook_context)

        assert session_ctx.task_count == 1
        assert session_ctx.successful_tasks == 0

    @pytest.mark.asyncio
    async def test_execute_output_credit_result(self, mock_memory_client, hook_context):
        """Test execution outputs credit result."""
        hook_context.metadata = {
            "task_id": "task-123",
            "success": True,
            "session_context": SessionContext(session_id="test"),
        }

        mock_memory_client.report_task_outcome.return_value = CreditAssignmentResult(
            credited=5,
            reconsolidated=["m1", "m2"],
            total_lr_applied=0.05,
        )

        hook = TaskOutcomeHook(memory_client=mock_memory_client)

        result = await hook.execute(hook_context)

        assert "credit_result" in result.output_data
        credit = result.output_data["credit_result"]
        assert credit["credited"] == 5
        assert credit["total_lr"] == 0.05


class TestIdleConsolidationHook:
    """Tests for IdleConsolidationHook."""

    def test_init_defaults(self):
        """Test default initialization."""
        hook = IdleConsolidationHook()
        assert hook.name == "idle_consolidation"
        assert hook._idle_threshold == 300.0
        assert hook._consolidation_mode == "light"

    def test_init_custom(self, mock_memory_client):
        """Test custom initialization."""
        hook = IdleConsolidationHook(
            memory_client=mock_memory_client,
            idle_threshold_seconds=60.0,
            consolidation_mode="deep",
        )
        assert hook._idle_threshold == 60.0
        assert hook._consolidation_mode == "deep"

    @pytest.mark.asyncio
    async def test_execute_first_call(self, hook_context):
        """Test first execution sets activity time."""
        hook = IdleConsolidationHook()

        await hook.execute(hook_context)

        assert hook._last_activity is not None

    @pytest.mark.asyncio
    async def test_execute_below_threshold(self, mock_memory_client, hook_context):
        """Test no consolidation below threshold."""
        hook = IdleConsolidationHook(
            memory_client=mock_memory_client,
            idle_threshold_seconds=300.0,
        )
        hook._last_activity = datetime.now()

        await hook.execute(hook_context)

        mock_memory_client.trigger_consolidation.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_above_threshold(self, mock_memory_client, hook_context):
        """Test consolidation above threshold."""
        hook = IdleConsolidationHook(
            memory_client=mock_memory_client,
            idle_threshold_seconds=60.0,
        )
        hook._last_activity = datetime.now() - timedelta(seconds=120)
        mock_memory_client.trigger_consolidation.return_value = {}

        result = await hook.execute(hook_context)

        mock_memory_client.trigger_consolidation.assert_called_once()
        assert "idle_consolidation" in result.output_data
        assert result.output_data["idle_consolidation"]["triggered"] is True

    def test_update_activity(self):
        """Test update_activity method."""
        hook = IdleConsolidationHook()
        old_time = datetime.now() - timedelta(hours=1)
        hook._last_activity = old_time

        hook.update_activity()

        assert hook._last_activity > old_time


class TestCreateSessionHooks:
    """Tests for create_session_hooks convenience function."""

    def test_creates_all_hooks(self, mock_memory_client):
        """Test creates all three hooks."""
        start, end, outcome = create_session_hooks(mock_memory_client)

        assert isinstance(start, SessionStartHook)
        assert isinstance(end, SessionEndHook)
        assert isinstance(outcome, TaskOutcomeHook)

    def test_all_hooks_share_memory_client(self, mock_memory_client):
        """Test all hooks share the same memory client."""
        start, end, outcome = create_session_hooks(mock_memory_client)

        assert start._memory == mock_memory_client
        assert end._memory == mock_memory_client
        assert outcome._memory == mock_memory_client

    def test_auto_consolidate_option(self, mock_memory_client):
        """Test auto_consolidate option."""
        _, end, _ = create_session_hooks(
            mock_memory_client,
            auto_consolidate=False,
        )

        assert end._auto_consolidate is False

    def test_load_context_option(self, mock_memory_client):
        """Test load_context option."""
        start, _, _ = create_session_hooks(
            mock_memory_client,
            load_context=False,
        )

        assert start._load_context is False
