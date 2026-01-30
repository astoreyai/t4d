"""
Core lifecycle hooks for World Weaver modules.

Provides hooks for:
- Module initialization
- Graceful shutdown
- Health checks
- Configuration changes
"""

import logging

from ww.hooks.base import Hook, HookContext, HookPriority

logger = logging.getLogger(__name__)


class CoreHook(Hook):
    """Base class for core lifecycle hooks."""

    def __init__(
        self,
        name: str,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
    ):
        super().__init__(name, priority, enabled)


class InitHook(CoreHook):
    """
    Hook executed during module initialization.

    Use for:
    - Resource allocation
    - Connection establishment
    - Cache warming
    - State restoration
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute initialization logic.

        Context data:
        - input_data["module_name"]: Name of module being initialized
        - input_data["config"]: Module configuration
        - input_data["session_id"]: Session identifier

        Returns:
            Context with initialization results
        """
        module_name = context.input_data.get("module_name", "unknown")
        logger.info(f"[{self.name}] Initializing module: {module_name}")
        return context


class ShutdownHook(CoreHook):
    """
    Hook executed during graceful shutdown.

    Use for:
    - Resource cleanup
    - Connection closing
    - State persistence
    - Flush buffers
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute shutdown logic.

        Context data:
        - input_data["module_name"]: Name of module being shut down
        - input_data["reason"]: Shutdown reason (optional)
        - input_data["timeout"]: Shutdown timeout in seconds

        Returns:
            Context with shutdown results
        """
        module_name = context.input_data.get("module_name", "unknown")
        reason = context.input_data.get("reason", "normal")
        logger.info(f"[{self.name}] Shutting down module: {module_name} (reason: {reason})")
        return context


class HealthCheckHook(CoreHook):
    """
    Hook executed during health checks.

    Use for:
    - Connection validation
    - Resource availability
    - Performance metrics
    - Diagnostic information
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute health check logic.

        Context data:
        - input_data["module_name"]: Name of module being checked
        - input_data["check_type"]: "liveness" or "readiness"

        Expected output:
        - output_data["status"]: "healthy", "degraded", or "unhealthy"
        - output_data["message"]: Status description
        - output_data["metrics"]: Optional performance metrics

        Returns:
            Context with health status
        """
        module_name = context.input_data.get("module_name", "unknown")
        check_type = context.input_data.get("check_type", "liveness")

        # Default to healthy
        context.set_result(
            status="healthy",
            message=f"{module_name} is operational",
            check_type=check_type,
        )

        return context


class ConfigChangeHook(CoreHook):
    """
    Hook executed when configuration changes.

    Use for:
    - Reloading settings
    - Adjusting resource allocations
    - Updating thresholds
    - Invalidating caches
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute configuration change logic.

        Context data:
        - input_data["module_name"]: Name of module
        - input_data["old_config"]: Previous configuration
        - input_data["new_config"]: New configuration
        - input_data["changed_keys"]: List of changed configuration keys

        Returns:
            Context with configuration update results
        """
        module_name = context.input_data.get("module_name", "unknown")
        changed_keys = context.input_data.get("changed_keys", [])

        logger.info(
            f"[{self.name}] Configuration changed for {module_name}: "
            f"{len(changed_keys)} keys updated"
        )

        return context


# Example implementations

class LoggingInitHook(InitHook):
    """Example: Log module initialization with timing."""

    def __init__(self):
        super().__init__(
            name="logging_init",
            priority=HookPriority.LOW,
        )

    async def execute(self, context: HookContext) -> HookContext:
        module = context.input_data.get("module_name", "unknown")
        logger.info(f"Initializing {module} at {context.start_time}")
        return context


class HealthMetricsHook(HealthCheckHook):
    """Example: Collect and report health metrics."""

    def __init__(self):
        super().__init__(
            name="health_metrics",
            priority=HookPriority.NORMAL,
        )

    async def execute(self, context: HookContext) -> HookContext:
        module = context.input_data.get("module_name", "unknown")

        # Example metrics collection
        metrics = {
            "uptime_ms": context.elapsed_ms(),
            "module": module,
            "timestamp": context.start_time.isoformat(),
        }

        context.metadata["health_metrics"] = metrics
        logger.debug(f"Health metrics for {module}: {metrics}")

        return context


class GracefulShutdownHook(ShutdownHook):
    """Example: Ensure graceful shutdown with timeout."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(
            name="graceful_shutdown",
            priority=HookPriority.CRITICAL,
        )
        self.timeout_seconds = timeout_seconds

    async def execute(self, context: HookContext) -> HookContext:
        import asyncio

        module = context.input_data.get("module_name", "unknown")
        timeout = context.input_data.get("timeout", self.timeout_seconds)

        logger.info(f"Graceful shutdown for {module} (timeout: {timeout}s)")

        # Allow some time for cleanup
        await asyncio.sleep(0.1)

        context.set_result(
            shutdown_complete=True,
            timeout_used=timeout,
        )

        return context


class ConfigValidationHook(ConfigChangeHook):
    """Example: Validate configuration changes before applying."""

    def __init__(self, validators: dict[str, callable] | None = None):
        super().__init__(
            name="config_validation",
            priority=HookPriority.CRITICAL,
        )
        self.validators = validators or {}

    async def execute(self, context: HookContext) -> HookContext:
        new_config = context.input_data.get("new_config", {})
        changed_keys = context.input_data.get("changed_keys", [])

        # Validate changed keys
        for key in changed_keys:
            if key in self.validators:
                validator = self.validators[key]
                value = new_config.get(key)

                try:
                    validator(value)
                except ValueError as e:
                    logger.error(f"Config validation failed for {key}: {e}")
                    context.set_error(
                        e,
                        key=key,
                        value=value,
                    )
                    raise

        logger.info(f"Configuration validated: {len(changed_keys)} keys")
        return context
