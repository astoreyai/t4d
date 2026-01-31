"""Tests for hooks base module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from t4dm.hooks.base import (
    Hook,
    HookContext,
    HookError,
    HookPhase,
    HookPriority,
    HookRegistry,
    with_hooks,
)


class TestHookPhase:
    """Tests for HookPhase enum."""

    def test_phases_exist(self):
        """Test all phases are defined."""
        assert HookPhase.PRE.value == "pre"
        assert HookPhase.ON.value == "on"
        assert HookPhase.POST.value == "post"
        assert HookPhase.ERROR.value == "error"


class TestHookPriority:
    """Tests for HookPriority enum."""

    def test_priorities_exist(self):
        """Test all priorities are defined."""
        assert HookPriority.CRITICAL.value == 0
        assert HookPriority.HIGH.value == 100
        assert HookPriority.NORMAL.value == 500
        assert HookPriority.LOW.value == 1000

    def test_priority_ordering(self):
        """Test priorities have correct ordering."""
        assert HookPriority.CRITICAL.value < HookPriority.HIGH.value
        assert HookPriority.HIGH.value < HookPriority.NORMAL.value
        assert HookPriority.NORMAL.value < HookPriority.LOW.value


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        ctx = HookContext()
        assert ctx.phase == HookPhase.PRE
        assert ctx.operation == ""
        assert ctx.input_data == {}
        assert ctx.output_data is None
        assert ctx.error is None
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.metadata == {}
        assert isinstance(ctx.start_time, datetime)
        assert ctx.hook_id is not None

    def test_elapsed_ms(self):
        """Test elapsed time calculation."""
        past = datetime.now() - timedelta(milliseconds=100)
        ctx = HookContext(start_time=past)
        elapsed = ctx.elapsed_ms()
        assert elapsed >= 100

    def test_set_error(self):
        """Test setting error."""
        ctx = HookContext()
        error = ValueError("test error")
        ctx.set_error(error, context_key="value")
        assert ctx.error == error
        assert ctx.error_context["context_key"] == "value"
        assert ctx.end_time is not None

    def test_set_result(self):
        """Test setting result."""
        ctx = HookContext()
        ctx.set_result(key="value", count=5)
        assert ctx.output_data == {"key": "value", "count": 5}
        assert ctx.end_time is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="test",
            session_id="session-123",
        )
        d = ctx.to_dict()
        assert d["phase"] == "pre"
        assert d["operation"] == "test"
        assert d["session_id"] == "session-123"
        assert "hook_id" in d
        assert "elapsed_ms" in d


class TestHookError:
    """Tests for HookError exception."""

    def test_initialization(self):
        """Test error initialization."""
        orig_error = ValueError("original")
        error = HookError(
            message="Hook failed",
            hook_name="test_hook",
            original_error=orig_error,
            context={"key": "value"},
        )
        assert str(error) == "Hook failed"
        assert error.hook_name == "test_hook"
        assert error.original_error is orig_error
        assert error.context == {"key": "value"}


class ConcreteHook(Hook):
    """Concrete hook implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execute_count = 0

    async def execute(self, context: HookContext) -> HookContext:
        self.execute_count += 1
        return context


class TestHook:
    """Tests for Hook base class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ConcreteHook(
            name="test_hook",
            priority=HookPriority.HIGH,
            enabled=True,
        )
        assert hook.name == "test_hook"
        assert hook.priority == HookPriority.HIGH
        assert hook.enabled is True

    def test_default_values(self):
        """Test default initialization values."""
        hook = ConcreteHook(name="test")
        assert hook.priority == HookPriority.NORMAL
        assert hook.enabled is True

    def test_should_execute_when_enabled(self):
        """Test should_execute returns True when enabled."""
        hook = ConcreteHook(name="test", enabled=True)
        ctx = HookContext()
        assert hook.should_execute(ctx) is True

    def test_should_execute_when_disabled(self):
        """Test should_execute returns False when disabled."""
        hook = ConcreteHook(name="test", enabled=False)
        ctx = HookContext()
        assert hook.should_execute(ctx) is False

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = ConcreteHook(name="test")
        ctx = HookContext()
        result = await hook.execute(ctx)
        assert result is ctx
        assert hook.execute_count == 1

    def test_get_stats(self):
        """Test getting hook statistics."""
        hook = ConcreteHook(name="test", priority=HookPriority.HIGH)
        stats = hook.get_stats()
        assert stats["name"] == "test"
        assert stats["priority"] == 100
        assert stats["enabled"] is True
        assert stats["executions"] == 0
        assert stats["errors"] == 0

    def test_increment_stats(self):
        """Test incrementing execution stats."""
        hook = ConcreteHook(name="test")
        hook.increment_stats(success=True)
        hook.increment_stats(success=False)
        assert hook._execution_count == 2
        assert hook._error_count == 1

    @pytest.mark.asyncio
    async def test_on_error(self):
        """Test error handling callback."""
        hook = ConcreteHook(name="test")
        ctx = HookContext()
        ctx.set_error(ValueError("test"))
        # Should not raise
        await hook.on_error(ctx)


class TestHookRegistry:
    """Tests for HookRegistry class."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = HookRegistry(name="test")
        assert registry.name == "test"
        assert len(registry._hooks[HookPhase.PRE]) == 0
        assert len(registry._hooks[HookPhase.POST]) == 0

    def test_register_hook(self):
        """Test registering a hook."""
        registry = HookRegistry()
        hook = ConcreteHook(name="test")
        registry.register(hook, HookPhase.PRE)
        assert len(registry._hooks[HookPhase.PRE]) == 1

    def test_register_different_phases(self):
        """Test registering hooks in different phases."""
        registry = HookRegistry()
        hook1 = ConcreteHook(name="h1")
        hook2 = ConcreteHook(name="h2")
        registry.register(hook1, HookPhase.PRE)
        registry.register(hook2, HookPhase.POST)

        assert len(registry._hooks[HookPhase.PRE]) == 1
        assert len(registry._hooks[HookPhase.POST]) == 1

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        registry = HookRegistry()
        hook = ConcreteHook(name="test")
        registry.register(hook, HookPhase.PRE)
        assert len(registry._hooks[HookPhase.PRE]) == 1

        registry.unregister("test")
        assert len(registry._hooks[HookPhase.PRE]) == 0

    def test_unregister_from_specific_phase(self):
        """Test unregistering from specific phase."""
        registry = HookRegistry()
        hook = ConcreteHook(name="test")
        registry.register(hook, HookPhase.PRE)
        registry.register(hook, HookPhase.POST)

        registry.unregister("test", HookPhase.PRE)
        assert len(registry._hooks[HookPhase.PRE]) == 0
        assert len(registry._hooks[HookPhase.POST]) == 1

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent hook."""
        registry = HookRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_hooks(self):
        """Test getting hooks by phase."""
        registry = HookRegistry()
        hook1 = ConcreteHook(name="h1")
        hook2 = ConcreteHook(name="h2")
        registry.register(hook1, HookPhase.PRE)
        registry.register(hook2, HookPhase.POST)

        pre_hooks = registry.get_hooks(HookPhase.PRE)
        assert len(pre_hooks) == 1
        assert pre_hooks[0].name == "h1"

    def test_get_hooks_sorted_by_priority(self):
        """Test hooks are sorted by priority."""
        registry = HookRegistry()
        low = ConcreteHook(name="low", priority=HookPriority.LOW)
        high = ConcreteHook(name="high", priority=HookPriority.HIGH)
        normal = ConcreteHook(name="normal", priority=HookPriority.NORMAL)

        registry.register(low, HookPhase.PRE)
        registry.register(high, HookPhase.PRE)
        registry.register(normal, HookPhase.PRE)

        hooks = registry.get_hooks(HookPhase.PRE)
        names = [h.name for h in hooks]
        # PRE hooks sorted ascending by priority value
        assert names == ["high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_execute_phase(self):
        """Test executing hooks for a phase."""
        registry = HookRegistry()
        hook = ConcreteHook(name="test")
        registry.register(hook, HookPhase.PRE)

        ctx = HookContext(operation="test")
        result = await registry.execute_phase(HookPhase.PRE, ctx)

        assert result.phase == HookPhase.PRE
        assert hook.execute_count == 1

    @pytest.mark.asyncio
    async def test_execute_phase_skips_disabled(self):
        """Test disabled hooks are skipped."""
        registry = HookRegistry()
        hook = ConcreteHook(name="test", enabled=False)
        registry.register(hook, HookPhase.ON)

        ctx = HookContext()
        await registry.execute_phase(HookPhase.ON, ctx)

        assert hook.execute_count == 0

    @pytest.mark.asyncio
    async def test_execute_phase_error_handling(self):
        """Test error handling during execution."""
        class FailingHook(Hook):
            async def execute(self, context):
                raise ValueError("hook failed")

        registry = HookRegistry()
        hook = FailingHook(name="failing")
        registry.register(hook, HookPhase.ON)

        ctx = HookContext()
        result = await registry.execute_phase(HookPhase.ON, ctx)

        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_phase_fail_fast(self):
        """Test fail_fast stops execution on error."""
        class FailingHook(Hook):
            async def execute(self, context):
                raise ValueError("hook failed")

        registry = HookRegistry(fail_fast=True)
        hook = FailingHook(name="failing")
        registry.register(hook, HookPhase.ON)

        ctx = HookContext()
        with pytest.raises(HookError):
            await registry.execute_phase(HookPhase.ON, ctx)

    def test_get_stats(self):
        """Test getting registry statistics."""
        registry = HookRegistry(name="test_registry")
        hook1 = ConcreteHook(name="h1", priority=HookPriority.HIGH)
        hook2 = ConcreteHook(name="h2", priority=HookPriority.LOW)
        registry.register(hook1, HookPhase.PRE)
        registry.register(hook2, HookPhase.POST)

        stats = registry.get_stats()
        assert stats["registry"] == "test_registry"
        assert stats["total_hooks"] == 2

    def test_clear(self):
        """Test clearing all hooks."""
        registry = HookRegistry()
        registry.register(ConcreteHook(name="h1"), HookPhase.PRE)
        registry.register(ConcreteHook(name="h2"), HookPhase.POST)

        registry.clear()

        for phase in HookPhase:
            assert len(registry._hooks[phase]) == 0


class TestWithHooksDecorator:
    """Tests for with_hooks decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test basic decorator functionality."""
        registry = HookRegistry()
        hook = ConcreteHook(name="test")
        registry.register(hook, HookPhase.PRE)

        @with_hooks(registry, "test_op")
        async def test_func():
            return "result"

        result = await test_func()
        assert result == "result"
        assert hook.execute_count >= 1

    @pytest.mark.asyncio
    async def test_decorator_pre_post_hooks(self):
        """Test pre and post hooks are called."""
        registry = HookRegistry()
        pre_hook = ConcreteHook(name="pre")
        post_hook = ConcreteHook(name="post")
        registry.register(pre_hook, HookPhase.PRE)
        registry.register(post_hook, HookPhase.POST)

        @with_hooks(registry, "test_op")
        async def test_func():
            return "done"

        await test_func()
        assert pre_hook.execute_count == 1
        assert post_hook.execute_count == 1

    @pytest.mark.asyncio
    async def test_decorator_with_error(self):
        """Test decorator handles function errors."""
        registry = HookRegistry()
        hook = ConcreteHook(name="test")
        registry.register(hook, HookPhase.PRE)

        @with_hooks(registry, "test_op")
        async def failing_func():
            raise ValueError("function failed")

        with pytest.raises(ValueError):
            await failing_func()
