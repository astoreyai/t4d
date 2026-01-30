"""
Tests for World Weaver hooks system.

Tests:
- Hook registration and execution
- Priority ordering
- Error isolation
- Parallel execution
- Context propagation
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from ww.hooks.base import (
    Hook,
    HookRegistry,
    HookContext,
    HookPhase,
    HookPriority,
    HookError,
    with_hooks,
)
from ww.hooks.core import InitHook, ShutdownHook, HealthCheckHook
from ww.hooks.memory import CreateHook, RecallHook, AccessHook
from ww.hooks.storage import QueryHook, ErrorHook, RetryHook


class TestHookContext:
    """Test HookContext functionality."""

    def test_context_creation(self):
        """Test context initialization."""
        context = HookContext(
            operation="test_op",
            module="test_module",
            session_id="test-session",
        )

        assert context.operation == "test_op"
        assert context.module == "test_module"
        assert context.session_id == "test-session"
        assert context.phase == HookPhase.PRE
        assert context.error is None

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        context = HookContext()
        asyncio.run(asyncio.sleep(0.01))
        elapsed = context.elapsed_ms()

        assert elapsed >= 10  # At least 10ms

    def test_set_error(self):
        """Test error state setting."""
        context = HookContext()
        error = ValueError("test error")

        context.set_error(error, field="test_field")

        assert context.error == error
        assert context.error_context["field"] == "test_field"
        assert context.end_time is not None

    def test_set_result(self):
        """Test result setting."""
        context = HookContext()
        context.set_result(status="success", count=42)

        assert context.output_data["status"] == "success"
        assert context.output_data["count"] == 42
        assert context.end_time is not None


class TestHookRegistry:
    """Test HookRegistry functionality."""

    def test_registry_creation(self):
        """Test registry initialization."""
        registry = HookRegistry(name="test")

        assert registry.name == "test"
        assert registry.fail_fast is False
        assert len(registry.get_hooks(HookPhase.PRE)) == 0

    def test_hook_registration(self):
        """Test hook registration."""
        registry = HookRegistry()

        class TestHook(Hook):
            async def execute(self, context):
                return context

        hook = TestHook(name="test_hook")
        registry.register(hook, HookPhase.PRE)

        hooks = registry.get_hooks(HookPhase.PRE)
        assert len(hooks) == 1
        assert hooks[0].name == "test_hook"

    def test_priority_ordering(self):
        """Test hooks are ordered by priority."""
        registry = HookRegistry()

        class TestHook(Hook):
            async def execute(self, context):
                return context

        # Register in reverse priority order
        hook_low = TestHook(name="low", priority=HookPriority.LOW)
        hook_critical = TestHook(name="critical", priority=HookPriority.CRITICAL)
        hook_normal = TestHook(name="normal", priority=HookPriority.NORMAL)

        registry.register(hook_low, HookPhase.PRE)
        registry.register(hook_critical, HookPhase.PRE)
        registry.register(hook_normal, HookPhase.PRE)

        hooks = registry.get_hooks(HookPhase.PRE)

        # PRE hooks should be sorted: CRITICAL, NORMAL, LOW
        assert hooks[0].name == "critical"
        assert hooks[1].name == "normal"
        assert hooks[2].name == "low"

    def test_duplicate_registration(self):
        """Test duplicate hook names are rejected."""
        registry = HookRegistry()

        class TestHook(Hook):
            async def execute(self, context):
                return context

        hook1 = TestHook(name="duplicate")
        hook2 = TestHook(name="duplicate")

        registry.register(hook1, HookPhase.PRE)
        registry.register(hook2, HookPhase.PRE)

        # Should only have one hook
        hooks = registry.get_hooks(HookPhase.PRE)
        assert len(hooks) == 1

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test hooks execute sequentially in order."""
        registry = HookRegistry()
        execution_order = []

        class OrderedHook(Hook):
            def __init__(self, name, priority):
                super().__init__(name, priority)

            async def execute(self, context):
                execution_order.append(self.name)
                return context

        registry.register(OrderedHook("first", HookPriority.CRITICAL), HookPhase.PRE)
        registry.register(OrderedHook("second", HookPriority.NORMAL), HookPhase.PRE)
        registry.register(OrderedHook("third", HookPriority.LOW), HookPhase.PRE)

        context = HookContext(operation="test")
        await registry.execute_phase(HookPhase.PRE, context)

        assert execution_order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """Test hook errors are isolated."""
        registry = HookRegistry(fail_fast=False)

        class FailingHook(Hook):
            async def execute(self, context):
                raise ValueError("Hook failed")

        class SuccessHook(Hook):
            async def execute(self, context):
                context.metadata["success"] = True
                return context

        registry.register(FailingHook(name="failing"), HookPhase.PRE)
        registry.register(SuccessHook(name="success"), HookPhase.PRE)

        context = HookContext(operation="test")
        await registry.execute_phase(HookPhase.PRE, context)

        # Success hook should still execute
        assert context.metadata.get("success") is True

    @pytest.mark.asyncio
    async def test_fail_fast(self):
        """Test fail_fast stops execution on error."""
        registry = HookRegistry(fail_fast=True)

        class FailingHook(Hook):
            async def execute(self, context):
                raise ValueError("Hook failed")

        class SuccessHook(Hook):
            async def execute(self, context):
                context.metadata["success"] = True
                return context

        registry.register(FailingHook(name="failing"), HookPhase.PRE)
        registry.register(SuccessHook(name="success"), HookPhase.PRE)

        context = HookContext(operation="test")

        with pytest.raises(HookError):
            await registry.execute_phase(HookPhase.PRE, context)

        # Success hook should not execute
        assert "success" not in context.metadata

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test POST hooks execute in parallel."""
        registry = HookRegistry()
        execution_times = {}

        class SlowHook(Hook):
            async def execute(self, context):
                import time
                start = time.time()
                await asyncio.sleep(0.1)
                execution_times[self.name] = time.time() - start
                return context

        registry.register(SlowHook(name="hook1"), HookPhase.POST)
        registry.register(SlowHook(name="hook2"), HookPhase.POST)
        registry.register(SlowHook(name="hook3"), HookPhase.POST)

        context = HookContext(operation="test")
        import time
        start = time.time()
        await registry.execute_phase(HookPhase.POST, context, parallel=True)
        total_time = time.time() - start

        # Parallel execution should be faster than sequential
        # 3 hooks * 0.1s = 0.3s sequential, but parallel should be ~0.1s
        assert total_time < 0.2  # Should be much less than 0.3s


class TestCoreHooks:
    """Test core lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_init_hook(self):
        """Test initialization hook."""
        class TestInitHook(InitHook):
            def __init__(self):
                super().__init__(name="test_init")
                self.initialized = False

            async def execute(self, context):
                self.initialized = True
                return await super().execute(context)

        hook = TestInitHook()
        context = HookContext(
            operation="initialize",
            input_data={"module_name": "test_module"},
        )

        await hook.execute(context)
        assert hook.initialized is True

    @pytest.mark.asyncio
    async def test_health_check_hook(self):
        """Test health check hook."""
        hook = HealthCheckHook(name="test_health")
        context = HookContext(
            operation="health_check",
            input_data={
                "module_name": "test_module",
                "check_type": "liveness",
            },
        )

        context = await hook.execute(context)

        assert context.output_data["status"] == "healthy"
        assert "message" in context.output_data


class TestMemoryHooks:
    """Test memory module hooks."""

    @pytest.mark.asyncio
    async def test_create_hook(self):
        """Test memory creation hook."""
        class TestCreateHook(CreateHook):
            def __init__(self):
                super().__init__(name="test_create")
                self.pre_called = False
                self.post_called = False

            async def execute(self, context):
                if context.phase == HookPhase.PRE:
                    self.pre_called = True
                elif context.phase == HookPhase.POST:
                    self.post_called = True
                return context

        hook = TestCreateHook()

        # Test PRE phase
        context_pre = HookContext(
            operation="create",
            phase=HookPhase.PRE,
            input_data={
                "memory_type": "episodic",
                "content": "test content",
            },
        )
        await hook.execute(context_pre)
        assert hook.pre_called is True

        # Test POST phase
        context_post = HookContext(
            operation="create",
            phase=HookPhase.POST,
            input_data={"memory_type": "episodic"},
        )
        context_post.set_result(memory_id="test-id")
        await hook.execute(context_post)
        assert hook.post_called is True

    @pytest.mark.asyncio
    async def test_memory_type_filter(self):
        """Test memory hook filtering by type."""
        hook = CreateHook(
            name="episodic_only",
            memory_type="episodic",
        )

        # Should execute for episodic
        context_episodic = HookContext(
            input_data={"memory_type": "episodic"},
        )
        assert hook.should_execute(context_episodic) is True

        # Should not execute for semantic
        context_semantic = HookContext(
            input_data={"memory_type": "semantic"},
        )
        assert hook.should_execute(context_semantic) is False


class TestStorageHooks:
    """Test storage module hooks."""

    @pytest.mark.asyncio
    async def test_query_hook(self):
        """Test query instrumentation hook."""
        class TimingHook(QueryHook):
            def __init__(self):
                super().__init__(name="timing")
                self.query_start = None
                self.duration = None

            async def execute(self, context):
                import time
                if context.phase == HookPhase.PRE:
                    self.query_start = time.time()
                elif context.phase == HookPhase.POST:
                    self.duration = time.time() - self.query_start
                return context

        hook = TimingHook()

        # PRE phase
        context = HookContext(
            phase=HookPhase.PRE,
            input_data={
                "storage_type": "neo4j",
                "query": "MATCH (n) RETURN n",
            },
        )
        await hook.execute(context)
        assert hook.query_start is not None

        # POST phase
        context.phase = HookPhase.POST
        await asyncio.sleep(0.01)
        await hook.execute(context)
        assert hook.duration >= 0.01


class TestHookDecorator:
    """Test with_hooks decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test basic decorator functionality."""
        registry = HookRegistry(name="test")

        class TestHook(Hook):
            def __init__(self):
                super().__init__(name="test")
                self.executed = False

            async def execute(self, context):
                self.executed = True
                return context

        hook = TestHook()
        registry.register(hook, HookPhase.PRE)

        @with_hooks(registry, operation="test_op", module="test")
        async def test_function(value: int) -> int:
            return value * 2

        result = await test_function(21)

        assert result == 42
        assert hook.executed is True

    @pytest.mark.asyncio
    async def test_decorator_error_handling(self):
        """Test decorator handles errors correctly."""
        registry = HookRegistry(name="test")

        class ErrorHook(Hook):
            def __init__(self):
                super().__init__(name="error")
                self.error_handled = False

            async def execute(self, context):
                # execute() is called for ERROR phase hooks when the main function fails
                self.error_handled = True
                return context

        hook = ErrorHook()
        registry.register(hook, HookPhase.ERROR)

        @with_hooks(registry, operation="test_op", module="test")
        async def failing_function():
            raise ValueError("Function failed")

        with pytest.raises(ValueError):
            await failing_function()

        assert hook.error_handled is True


class TestHookStatistics:
    """Test hook execution statistics."""

    @pytest.mark.asyncio
    async def test_execution_count(self):
        """Test execution count tracking."""
        # Create concrete implementation since Hook is abstract
        class TestHook(Hook):
            async def execute(self, context):
                return context

        hook = TestHook(name="test", priority=HookPriority.NORMAL)
        context = HookContext()

        for _ in range(5):
            await hook.execute(context)
            hook.increment_stats(success=True)

        stats = hook.get_stats()
        assert stats["executions"] == 5
        assert stats["errors"] == 0
        assert stats["error_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_error_rate(self):
        """Test error rate calculation."""
        # Create concrete implementation since Hook is abstract
        class TestHook(Hook):
            async def execute(self, context):
                return context

        hook = TestHook(name="test", priority=HookPriority.NORMAL)
        context = HookContext()

        # 3 successes, 2 failures
        for _ in range(3):
            await hook.execute(context)
            hook.increment_stats(success=True)

        for _ in range(2):
            hook.increment_stats(success=False)

        stats = hook.get_stats()
        assert stats["executions"] == 5
        assert stats["errors"] == 2
        assert stats["error_rate"] == 0.4

    def test_registry_stats(self):
        """Test registry statistics."""
        registry = HookRegistry(name="test")

        class TestHook(Hook):
            async def execute(self, context):
                return context

        registry.register(TestHook(name="hook1"), HookPhase.PRE)
        registry.register(TestHook(name="hook2"), HookPhase.POST)
        registry.register(TestHook(name="hook3"), HookPhase.POST)

        stats = registry.get_stats()

        assert stats["registry"] == "test"
        assert stats["total_hooks"] == 3
        assert stats["hooks_by_phase"]["pre"] == 1
        assert stats["hooks_by_phase"]["post"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
