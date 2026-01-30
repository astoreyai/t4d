"""
Base hook infrastructure for World Weaver.

Provides abstract base classes, execution engine, and error handling.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HookPhase(str, Enum):
    """Hook execution phase."""
    PRE = "pre"
    POST = "post"
    ON = "on"
    ERROR = "error"


class HookPriority(int, Enum):
    """
    Hook execution priority.

    Lower numbers execute first for PRE hooks.
    Higher numbers execute first for POST hooks.
    """
    CRITICAL = 0      # Security, validation, auth
    HIGH = 100        # Observability, auditing
    NORMAL = 500      # Business logic
    LOW = 1000        # Caching, cleanup


class HookError(Exception):
    """Exception raised during hook execution."""

    def __init__(
        self,
        message: str,
        hook_name: str,
        original_error: Exception | None = None,
        context: dict | None = None,
    ):
        super().__init__(message)
        self.hook_name = hook_name
        self.original_error = original_error
        self.context = context or {}


@dataclass
class HookContext:
    """
    Context passed to hook execution.

    Contains:
    - Operation metadata
    - Input/output data
    - Session/user context
    - Timing information
    - Error state
    """
    # Identifiers
    hook_id: UUID = field(default_factory=uuid4)
    session_id: str | None = None
    user_id: str | None = None

    # Operation context
    operation: str = ""
    phase: HookPhase = HookPhase.PRE
    module: str = ""

    # Data
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] | None = None

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Error tracking
    error: Exception | None = None
    error_context: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    def set_error(self, error: Exception, **context):
        """Set error state with context."""
        self.error = error
        self.error_context = context
        self.end_time = datetime.now()

    def set_result(self, **output_data):
        """Set successful result data."""
        self.output_data = output_data
        self.end_time = datetime.now()

    def to_dict(self) -> dict:
        """Serialize context for logging/tracing."""
        return {
            "hook_id": str(self.hook_id),
            "session_id": self.session_id,
            "operation": self.operation,
            "phase": self.phase.value,
            "module": self.module,
            "elapsed_ms": self.elapsed_ms(),
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata,
        }


class Hook(ABC):
    """
    Abstract base class for all hooks.

    Subclasses implement specific hook types (core, memory, storage, etc.)
    and provide execution logic for different phases.
    """

    def __init__(
        self,
        name: str,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
    ):
        """
        Initialize hook.

        Args:
            name: Unique hook identifier
            priority: Execution priority
            enabled: Whether hook is active
        """
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self._execution_count = 0
        self._error_count = 0

    @abstractmethod
    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute hook logic.

        Args:
            context: Hook execution context

        Returns:
            Modified context (can mutate input_data/output_data)

        Raises:
            HookError: If execution fails critically
        """

    async def on_error(self, context: HookContext) -> None:
        """
        Handle errors from other hooks (optional).

        Args:
            context: Context with error information
        """

    def should_execute(self, context: HookContext) -> bool:
        """
        Determine if hook should execute for given context.

        Override to implement conditional execution.

        Args:
            context: Hook execution context

        Returns:
            True if hook should execute
        """
        return self.enabled

    def increment_stats(self, success: bool = True) -> None:
        """Update execution statistics."""
        self._execution_count += 1
        if not success:
            self._error_count += 1

    def get_stats(self) -> dict:
        """Get hook execution statistics."""
        return {
            "name": self.name,
            "priority": self.priority.value,
            "enabled": self.enabled,
            "executions": self._execution_count,
            "errors": self._error_count,
            "error_rate": (
                self._error_count / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, priority={self.priority})"


class HookRegistry:
    """
    Registry for managing and executing hooks.

    Features:
    - Multi-phase hook execution (pre, post, on, error)
    - Priority-based ordering
    - Error isolation (one hook failure doesn't stop others)
    - Async execution with concurrency control
    - Performance metrics
    """

    def __init__(
        self,
        name: str = "default",
        fail_fast: bool = False,
        max_concurrent: int = 10,
    ):
        """
        Initialize hook registry.

        Args:
            name: Registry identifier
            fail_fast: Stop execution on first hook error
            max_concurrent: Maximum concurrent hook executions
        """
        self.name = name
        self.fail_fast = fail_fast
        self.max_concurrent = max_concurrent

        # Hooks organized by phase
        self._hooks: dict[HookPhase, list[Hook]] = {
            phase: [] for phase in HookPhase
        }

        # Execution semaphore
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def register(self, hook: Hook, phase: HookPhase = HookPhase.PRE) -> None:
        """
        Register a hook for a specific phase.

        Args:
            hook: Hook instance
            phase: Execution phase
        """
        hooks = self._hooks[phase]

        # Check for duplicate names
        if any(h.name == hook.name for h in hooks):
            logger.warning(
                f"Hook '{hook.name}' already registered in {phase}, skipping"
            )
            return

        hooks.append(hook)

        # Sort by priority (ascending for PRE, descending for POST)
        if phase == HookPhase.PRE:
            hooks.sort(key=lambda h: h.priority.value)
        else:
            hooks.sort(key=lambda h: -h.priority.value)

        logger.info(f"Registered {hook} in phase {phase}")

    def unregister(self, hook_name: str, phase: HookPhase | None = None) -> bool:
        """
        Unregister hook by name.

        Args:
            hook_name: Name of hook to remove
            phase: Specific phase, or None for all phases

        Returns:
            True if hook was removed
        """
        removed = False
        phases = [phase] if phase else list(HookPhase)

        for p in phases:
            hooks = self._hooks[p]
            initial_len = len(hooks)
            self._hooks[p] = [h for h in hooks if h.name != hook_name]
            if len(self._hooks[p]) < initial_len:
                removed = True
                logger.info(f"Unregistered hook '{hook_name}' from phase {p}")

        return removed

    def get_hooks(self, phase: HookPhase) -> list[Hook]:
        """Get all hooks for a phase."""
        return self._hooks[phase]

    async def execute_phase(
        self,
        phase: HookPhase,
        context: HookContext,
        parallel: bool = False,
    ) -> HookContext:
        """
        Execute all hooks for a phase.

        Args:
            phase: Hook phase to execute
            context: Execution context
            parallel: Execute hooks concurrently (use for independent hooks only)

        Returns:
            Modified context

        Raises:
            HookError: If fail_fast=True and a hook fails
        """
        hooks = self._hooks[phase]
        context.phase = phase

        if not hooks:
            return context

        logger.debug(f"Executing {len(hooks)} {phase} hooks for {context.operation}")

        if parallel and phase in (HookPhase.POST, HookPhase.ON):
            # Execute POST/ON hooks in parallel (order doesn't matter)
            return await self._execute_parallel(hooks, context)
        # Execute PRE hooks sequentially (order matters)
        return await self._execute_sequential(hooks, context)

    async def _execute_sequential(
        self,
        hooks: list[Hook],
        context: HookContext,
    ) -> HookContext:
        """Execute hooks sequentially with error handling."""
        errors = []

        for hook in hooks:
            if not hook.should_execute(context):
                continue

            try:
                async with self._semaphore:
                    context = await hook.execute(context)
                    hook.increment_stats(success=True)
            except Exception as e:
                hook.increment_stats(success=False)
                error = HookError(
                    message=f"Hook '{hook.name}' failed: {e}",
                    hook_name=hook.name,
                    original_error=e,
                    context=context.to_dict(),
                )
                errors.append(error)

                logger.error(f"Hook error: {error}", exc_info=e)

                if self.fail_fast:
                    raise error

        # Execute error hooks if any failures
        if errors:
            context.error = errors[0].original_error
            await self._execute_error_hooks(context)

        return context

    async def _execute_parallel(
        self,
        hooks: list[Hook],
        context: HookContext,
    ) -> HookContext:
        """Execute hooks in parallel with error isolation."""

        async def run_hook(hook: Hook) -> HookError | None:
            """Run single hook with error capture."""
            if not hook.should_execute(context):
                return None

            try:
                async with self._semaphore:
                    await hook.execute(context)
                    hook.increment_stats(success=True)
                    return None
            except Exception as e:
                hook.increment_stats(success=False)
                return HookError(
                    message=f"Hook '{hook.name}' failed: {e}",
                    hook_name=hook.name,
                    original_error=e,
                    context=context.to_dict(),
                )

        # Execute all hooks concurrently
        results = await asyncio.gather(
            *[run_hook(h) for h in hooks],
            return_exceptions=True,
        )

        # Collect errors
        errors = [r for r in results if isinstance(r, HookError)]

        if errors:
            for error in errors:
                logger.error(f"Hook error: {error}")

            context.error = errors[0].original_error
            await self._execute_error_hooks(context)

            if self.fail_fast:
                raise errors[0]

        return context

    async def _execute_error_hooks(self, context: HookContext) -> None:
        """Execute error phase hooks."""
        error_hooks = self._hooks[HookPhase.ERROR]

        for hook in error_hooks:
            try:
                await hook.on_error(context)
            except Exception as e:
                logger.error(
                    f"Error hook '{hook.name}' failed: {e}",
                    exc_info=True,
                )

    def get_stats(self) -> dict:
        """Get registry execution statistics."""
        all_hooks = []
        for phase, hooks in self._hooks.items():
            all_hooks.extend(hooks)

        return {
            "registry": self.name,
            "total_hooks": len(all_hooks),
            "hooks_by_phase": {
                phase.value: len(hooks)
                for phase, hooks in self._hooks.items()
            },
            "hook_stats": [h.get_stats() for h in all_hooks],
        }

    def clear(self) -> None:
        """Remove all registered hooks."""
        for phase in HookPhase:
            self._hooks[phase].clear()
        logger.info(f"Cleared all hooks from registry '{self.name}'")


# Decorator for adding hooks to functions
def with_hooks(
    registry: HookRegistry,
    operation: str,
    module: str = "",
    parallel_post: bool = True,
) -> Callable:
    """
    Decorator to wrap function with hook execution.

    Args:
        registry: Hook registry to use
        operation: Operation name for context
        module: Module name for context
        parallel_post: Execute POST hooks in parallel

    Returns:
        Decorated async function

    Example:
        @with_hooks(registry, "create_episode", "episodic")
        async def create(content: str, **kwargs) -> Episode:
            ...
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            # Create context
            context = HookContext(
                operation=operation,
                module=module,
                session_id=kwargs.get("session_id"),
                input_data={
                    "args": args,
                    "kwargs": kwargs,
                },
            )

            try:
                # PRE hooks (sequential, can modify input)
                context = await registry.execute_phase(HookPhase.PRE, context)

                # Extract potentially modified kwargs
                modified_kwargs = context.input_data.get("kwargs", kwargs)

                # Execute function
                result = await func(*args, **modified_kwargs)

                # Set result in context
                context.set_result(result=result)

                # POST hooks (can be parallel)
                context = await registry.execute_phase(
                    HookPhase.POST,
                    context,
                    parallel=parallel_post,
                )

                return result

            except Exception as e:
                context.set_error(e, function=func.__name__)

                # ERROR hooks
                await registry.execute_phase(HookPhase.ERROR, context)

                raise

        return wrapper
    return decorator
