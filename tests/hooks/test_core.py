"""Tests for core hooks module."""

import pytest
from datetime import datetime

from t4dm.hooks.core import (
    InitHook,
    ShutdownHook,
    HealthCheckHook,
    ConfigChangeHook,
    LoggingInitHook,
    HealthMetricsHook,
    GracefulShutdownHook,
    ConfigValidationHook,
    CoreHook,
)
from t4dm.hooks.base import HookContext, HookPhase, HookPriority


class TestInitHook:
    """Tests for InitHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = InitHook(name="init_test")
        assert hook.name == "init_test"
        assert hook.enabled is True

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = InitHook(name="init_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="init",
            input_data={"component": "test_component"},
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestShutdownHook:
    """Tests for ShutdownHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ShutdownHook(name="shutdown_test")
        assert hook.name == "shutdown_test"
        assert hook.enabled is True

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = ShutdownHook(name="shutdown_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="shutdown",
            input_data={"reason": "normal"},
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_with_graceful(self):
        """Test execution with graceful shutdown flag."""
        hook = ShutdownHook(name="shutdown_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="shutdown",
            input_data={"graceful": True, "timeout_seconds": 30},
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestHealthCheckHook:
    """Tests for HealthCheckHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = HealthCheckHook(name="health_test")
        assert hook.name == "health_test"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = HealthCheckHook(name="health_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="health_check",
            input_data={"component": "memory_system"},
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_with_component_status(self):
        """Test execution with component status data."""
        hook = HealthCheckHook(name="health_test")
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="health_check",
            output_data={"healthy": True, "latency_ms": 10},
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestConfigChangeHook:
    """Tests for ConfigChangeHook class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ConfigChangeHook(name="config_test")
        assert hook.name == "config_test"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = ConfigChangeHook(name="config_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="config_change",
            input_data={
                "key": "memory.decay_rate",
                "old_value": 0.1,
                "new_value": 0.2,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_pre_phase(self):
        """Test execution in pre phase for validation."""
        hook = ConfigChangeHook(name="config_test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="config_change",
            input_data={
                "key": "memory.max_size",
                "new_value": 1000,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestCoreHook:
    """Tests for CoreHook base class using concrete subclass."""

    def test_initialization_defaults(self):
        """Test default initialization via concrete subclass."""
        # Use InitHook as concrete implementation
        hook = InitHook(name="core_test")
        assert hook.name == "core_test"
        assert hook.priority == HookPriority.NORMAL
        assert hook.enabled is True

    def test_initialization_custom(self):
        """Test custom initialization via concrete subclass."""
        hook = InitHook(
            name="custom_core",
            priority=HookPriority.HIGH,
            enabled=False,
        )
        assert hook.name == "custom_core"
        assert hook.priority == HookPriority.HIGH
        assert hook.enabled is False


class TestLoggingInitHook:
    """Tests for LoggingInitHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = LoggingInitHook()
        assert hook.name == "logging_init"
        assert hook.priority == HookPriority.LOW

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = LoggingInitHook()
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="init",
            input_data={"module_name": "memory_system"},
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestHealthMetricsHook:
    """Tests for HealthMetricsHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = HealthMetricsHook()
        assert hook.name == "health_metrics"
        assert hook.priority == HookPriority.NORMAL

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = HealthMetricsHook()
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="health_check",
            input_data={"module_name": "episodic_memory"},
        )
        result = await hook.execute(ctx)
        assert result is ctx
        # Should have added metrics to metadata
        assert "health_metrics" in ctx.metadata
        metrics = ctx.metadata["health_metrics"]
        assert "uptime_ms" in metrics
        assert metrics["module"] == "episodic_memory"


class TestGracefulShutdownHook:
    """Tests for GracefulShutdownHook example implementation."""

    def test_initialization_default(self):
        """Test default initialization."""
        hook = GracefulShutdownHook()
        assert hook.name == "graceful_shutdown"
        assert hook.priority == HookPriority.CRITICAL
        assert hook.timeout_seconds == 30

    def test_initialization_custom_timeout(self):
        """Test custom timeout initialization."""
        hook = GracefulShutdownHook(timeout_seconds=60)
        assert hook.timeout_seconds == 60

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test hook execution."""
        hook = GracefulShutdownHook(timeout_seconds=1)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="shutdown",
            input_data={"module_name": "memory_system"},
        )
        result = await hook.execute(ctx)
        assert result is ctx
        assert ctx.output_data.get("shutdown_complete") is True

    @pytest.mark.asyncio
    async def test_execute_with_input_timeout(self):
        """Test execution uses input timeout if provided."""
        hook = GracefulShutdownHook(timeout_seconds=30)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="shutdown",
            input_data={"module_name": "test", "timeout": 5},
        )
        result = await hook.execute(ctx)
        assert ctx.output_data.get("timeout_used") == 5


class TestConfigValidationHook:
    """Tests for ConfigValidationHook example implementation."""

    def test_initialization_default(self):
        """Test default initialization."""
        hook = ConfigValidationHook()
        assert hook.name == "config_validation"
        assert hook.priority == HookPriority.CRITICAL
        assert hook.validators == {}

    def test_initialization_with_validators(self):
        """Test initialization with validators."""
        validators = {"max_size": lambda x: x > 0}
        hook = ConfigValidationHook(validators=validators)
        assert "max_size" in hook.validators

    @pytest.mark.asyncio
    async def test_execute_no_validators(self):
        """Test execution without validators."""
        hook = ConfigValidationHook()
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="config_change",
            input_data={
                "new_config": {"max_size": 100},
                "changed_keys": ["max_size"],
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_with_valid_config(self):
        """Test execution with valid config."""
        def validate_positive(x):
            if x <= 0:
                raise ValueError("Must be positive")

        hook = ConfigValidationHook(validators={"max_size": validate_positive})
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="config_change",
            input_data={
                "new_config": {"max_size": 100},
                "changed_keys": ["max_size"],
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_with_invalid_config(self):
        """Test execution with invalid config raises."""
        def validate_positive(x):
            if x <= 0:
                raise ValueError("Must be positive")

        hook = ConfigValidationHook(validators={"max_size": validate_positive})
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="config_change",
            input_data={
                "new_config": {"max_size": -1},
                "changed_keys": ["max_size"],
            },
        )
        with pytest.raises(ValueError, match="Must be positive"):
            await hook.execute(ctx)
