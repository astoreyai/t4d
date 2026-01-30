"""Tests for memory hooks module."""

import pytest
from datetime import datetime

from ww.hooks.memory import (
    CreateHook,
    RecallHook,
    UpdateHook,
    AccessHook,
    DecayHook,
    CachingRecallHook,
    ValidationHook,
)
from ww.hooks.base import HookContext, HookPhase, HookPriority


class TestCreateHook:
    """Tests for CreateHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = CreateHook(name="create_test")
        assert hook.name == "create_test"
        assert hook.enabled is True

    @pytest.mark.asyncio
    async def test_execute_pre(self):
        """Test pre-create execution."""
        hook = CreateHook(name="create_test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={
                "memory_type": "episodic",
                "content": "Test memory content",
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_post(self):
        """Test post-create execution."""
        hook = CreateHook(name="create_test")
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="create",
            input_data={"memory_type": "episodic"},
            output_data={"memory_id": "uuid-123"},
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestRecallHook:
    """Tests for RecallHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = RecallHook(name="recall_test")
        assert hook.name == "recall_test"

    @pytest.mark.asyncio
    async def test_execute_pre(self):
        """Test pre-recall execution."""
        hook = RecallHook(name="recall_test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="recall",
            input_data={
                "query": "test query",
                "limit": 10,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_post(self):
        """Test post-recall execution."""
        hook = RecallHook(name="recall_test")
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="recall",
            output_data={
                "results": [{"id": "1", "score": 0.9}],
                "count": 1,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestUpdateHook:
    """Tests for UpdateHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = UpdateHook(name="update_test")
        assert hook.name == "update_test"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test update hook execution."""
        hook = UpdateHook(name="update_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="update",
            input_data={
                "memory_id": "uuid-123",
                "changes": {"stability": 2.0},
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestAccessHook:
    """Tests for AccessHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = AccessHook(name="access_test")
        assert hook.name == "access_test"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test access hook execution."""
        hook = AccessHook(name="access_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="access",
            input_data={
                "memory_id": "uuid-123",
                "access_type": "read",
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestDecayHook:
    """Tests for DecayHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = DecayHook(name="decay_test")
        assert hook.name == "decay_test"

    @pytest.mark.asyncio
    async def test_execute_pre(self):
        """Test pre-decay execution."""
        hook = DecayHook(name="decay_test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="decay",
            input_data={
                "memory_id": "uuid-123",
                "current_stability": 2.0,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_post(self):
        """Test post-decay execution."""
        hook = DecayHook(name="decay_test")
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="decay",
            output_data={
                "new_stability": 1.8,
                "retrievability": 0.75,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestCachingRecallHook:
    """Tests for CachingRecallHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = CachingRecallHook()
        assert hook.name == "caching_recall"
        assert hook.priority == HookPriority.HIGH
        assert hook.cache == {}
        assert hook.cache_size == 1000

    def test_custom_cache_size(self):
        """Test custom cache size setting."""
        hook = CachingRecallHook(cache_size=500)
        assert hook.cache_size == 500

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss behavior."""
        hook = CachingRecallHook()
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="recall",
            input_data={"query": "test query", "memory_type": "episodic"},
        )
        result = await hook.execute(ctx)
        assert result.metadata.get("cache_hit") is False

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit behavior."""
        hook = CachingRecallHook()

        # First call - cache miss
        ctx1 = HookContext(
            phase=HookPhase.PRE,
            operation="recall",
            input_data={"query": "test query", "memory_type": "episodic"},
        )
        await hook.execute(ctx1)

        # Simulate post-phase storing result
        ctx2 = HookContext(
            phase=HookPhase.POST,
            operation="recall",
            input_data={"query": "test query", "memory_type": "episodic"},
            output_data={"results": [{"id": "1"}]},
            metadata={"cache_hit": False},
        )
        await hook.execute(ctx2)

        # Second call - cache hit
        ctx3 = HookContext(
            phase=HookPhase.PRE,
            operation="recall",
            input_data={"query": "test query", "memory_type": "episodic"},
        )
        result = await hook.execute(ctx3)
        assert result.metadata.get("cache_hit") is True

    def test_get_stats(self):
        """Test getting cache statistics."""
        hook = CachingRecallHook()
        hook.hits = 10
        hook.misses = 5
        stats = hook.get_stats()
        assert "cache" in stats
        assert stats["cache"]["hits"] == 10
        assert stats["cache"]["misses"] == 5


class TestValidationHook:
    """Tests for ValidationHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ValidationHook()
        assert hook.name == "validation"
        assert hook.priority == HookPriority.CRITICAL
        assert hook.max_content_length == 100000

    def test_custom_max_length(self):
        """Test custom max content length."""
        hook = ValidationHook(max_content_length=5000)
        assert hook.max_content_length == 5000

    @pytest.mark.asyncio
    async def test_valid_content(self):
        """Test validation passes for valid content."""
        hook = ValidationHook()
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"content": "Valid memory content"},
        )
        result = await hook.execute(ctx)
        assert result.error is None

    @pytest.mark.asyncio
    async def test_content_too_long(self):
        """Test validation fails for content exceeding max length."""
        hook = ValidationHook(max_content_length=100)
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"content": "x" * 101},
        )
        with pytest.raises(ValueError, match="Content too long"):
            await hook.execute(ctx)

    @pytest.mark.asyncio
    async def test_required_field_missing(self):
        """Test validation fails for missing required field."""
        hook = ValidationHook(required_fields=["session_id"])
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"content": "test"},
        )
        with pytest.raises(ValueError, match="Required field missing"):
            await hook.execute(ctx)

    @pytest.mark.asyncio
    async def test_required_field_present(self):
        """Test validation passes when required field is present."""
        hook = ValidationHook(required_fields=["session_id"])
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"content": "test", "session_id": "s123"},
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_skip_non_pre_phase(self):
        """Test validation is skipped for non-pre phases."""
        hook = ValidationHook(max_content_length=10)
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="create",
            input_data={"content": "x" * 1000},  # Would fail in pre-phase
        )
        result = await hook.execute(ctx)
        assert result is ctx  # No error


class TestAuditTrailHook:
    """Tests for AuditTrailHook example implementation."""

    def test_initialization_default(self):
        """Test hook initialization with default audit store."""
        from ww.hooks.memory import AuditTrailHook
        hook = AuditTrailHook()
        assert hook.name == "audit_trail"
        assert hook.priority == HookPriority.HIGH
        assert isinstance(hook.audit_store, list)

    def test_initialization_custom_store(self):
        """Test hook initialization with custom audit store."""
        from ww.hooks.memory import AuditTrailHook
        custom_store = {"entries": []}  # Use dict to ensure identity check works
        hook = AuditTrailHook(audit_store=custom_store)
        assert hook.audit_store is custom_store

    @pytest.mark.asyncio
    async def test_execute_records_audit(self):
        """Test execute records audit entry."""
        from ww.hooks.memory import AuditTrailHook
        hook = AuditTrailHook()
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"memory_type": "episodic"},
            session_id="session-123",
            user_id="user-456",
        )
        await hook.execute(ctx)
        assert len(hook.audit_store) == 1
        entry = hook.audit_store[0]
        assert entry["operation"] == "create"
        assert entry["memory_type"] == "episodic"
        assert entry["session_id"] == "session-123"
        assert entry["user_id"] == "user-456"
        assert entry["success"] is True

    @pytest.mark.asyncio
    async def test_execute_records_multiple_operations(self):
        """Test execute records multiple operations."""
        from ww.hooks.memory import AuditTrailHook
        hook = AuditTrailHook()

        for op in ["create", "recall", "update"]:
            ctx = HookContext(
                phase=HookPhase.ON,
                operation=op,
                input_data={"memory_type": "semantic"},
            )
            await hook.execute(ctx)

        assert len(hook.audit_store) == 3

    @pytest.mark.asyncio
    async def test_execute_records_error_state(self):
        """Test execute records error state."""
        from ww.hooks.memory import AuditTrailHook
        hook = AuditTrailHook()
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="create",
            input_data={"memory_type": "episodic"},
            error=ValueError("test error"),
        )
        await hook.execute(ctx)
        assert hook.audit_store[0]["success"] is False


class TestHebbianUpdateHook:
    """Tests for HebbianUpdateHook example implementation."""

    def test_initialization_default(self):
        """Test hook initialization with defaults."""
        from ww.hooks.memory import HebbianUpdateHook
        hook = HebbianUpdateHook()
        assert hook.name == "hebbian_update"
        assert hook.priority == HookPriority.NORMAL
        assert hook.learning_rate == 0.1

    def test_initialization_custom_rate(self):
        """Test hook initialization with custom learning rate."""
        from ww.hooks.memory import HebbianUpdateHook
        hook = HebbianUpdateHook(learning_rate=0.05)
        assert hook.learning_rate == 0.05

    @pytest.mark.asyncio
    async def test_execute_no_context_ids(self):
        """Test execute with no co-accessed memories."""
        from ww.hooks.memory import HebbianUpdateHook
        hook = HebbianUpdateHook()
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="access",
            input_data={
                "memory_id": "uuid-123",
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx
        assert "hebbian_updates" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_execute_with_context_ids(self):
        """Test execute with co-accessed memories."""
        from ww.hooks.memory import HebbianUpdateHook
        hook = HebbianUpdateHook()
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="access",
            input_data={
                "memory_id": "uuid-123",
                "context_ids": ["uuid-456", "uuid-789"],
            },
        )
        result = await hook.execute(ctx)
        assert ctx.metadata.get("hebbian_updates") == 2

    @pytest.mark.asyncio
    async def test_execute_with_empty_context_ids(self):
        """Test execute with empty context_ids list."""
        from ww.hooks.memory import HebbianUpdateHook
        hook = HebbianUpdateHook()
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="access",
            input_data={
                "memory_id": "uuid-123",
                "context_ids": [],
            },
        )
        result = await hook.execute(ctx)
        assert "hebbian_updates" not in ctx.metadata


class TestMemoryHookFiltering:
    """Tests for memory type filtering in MemoryHook."""

    def test_should_execute_no_filter(self):
        """Test should_execute with no memory type filter."""
        hook = CreateHook(name="test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"memory_type": "episodic"},
        )
        assert hook.should_execute(ctx) is True

    def test_should_execute_matching_filter(self):
        """Test should_execute with matching memory type filter."""
        hook = CreateHook(name="test", memory_type="episodic")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"memory_type": "episodic"},
        )
        assert hook.should_execute(ctx) is True

    def test_should_execute_non_matching_filter(self):
        """Test should_execute with non-matching memory type filter."""
        hook = CreateHook(name="test", memory_type="episodic")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={"memory_type": "semantic"},
        )
        assert hook.should_execute(ctx) is False

    def test_should_execute_disabled(self):
        """Test should_execute when hook is disabled."""
        hook = CreateHook(name="test", enabled=False)
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="create",
            input_data={},
        )
        assert hook.should_execute(ctx) is False


class TestCachingRecallHookAdvanced:
    """Additional tests for CachingRecallHook."""

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test cache evicts when full."""
        hook = CachingRecallHook(cache_size=2)

        # Add 3 items
        for i in range(3):
            ctx_pre = HookContext(
                phase=HookPhase.PRE,
                operation="recall",
                input_data={"query": f"query{i}", "memory_type": "episodic"},
            )
            await hook.execute(ctx_pre)

            ctx_post = HookContext(
                phase=HookPhase.POST,
                operation="recall",
                input_data={"query": f"query{i}", "memory_type": "episodic"},
                output_data={"results": [{"id": str(i)}]},
                metadata={"cache_hit": False},
            )
            await hook.execute(ctx_post)

        # Cache should only have 2 items
        assert len(hook.cache) <= 2

    def test_get_stats_zero_total(self):
        """Test get_stats with zero hits and misses."""
        hook = CachingRecallHook()
        stats = hook.get_stats()
        assert stats["cache"]["hit_rate"] == 0.0
