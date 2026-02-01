"""
Comprehensive unit tests for World Weaver observability module.

Tests structured logging, metrics collection, health checks, and timers
with 90%+ target coverage.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from t4dm.observability.logging import (
    LogContext,
    StructuredFormatter,
    ContextAdapter,
    configure_logging,
    get_logger,
    set_context,
    clear_context,
    OperationLogger,
    log_operation,
)
from t4dm.observability.metrics import (
    OperationMetrics,
    MetricsCollector,
    get_metrics,
    timed_operation,
    count_operation,
    Timer,
    AsyncTimer,
)
from t4dm.observability.health import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthChecker,
    get_health_checker,
)


# =============================================================================
# Structured Logging Tests
# =============================================================================


def test_log_context_creation():
    """Test LogContext dataclass creation."""
    ctx = LogContext(
        level="INFO",
        message="Test message",
        session_id="session-123",
        operation_id="op-456",
    )

    assert ctx.level == "INFO"
    assert ctx.message == "Test message"
    assert ctx.session_id == "session-123"
    assert ctx.operation_id == "op-456"


def test_log_context_to_json():
    """Test LogContext converts to valid JSON."""
    ctx = LogContext(
        level="WARNING",
        message="Warning message",
        session_id="sess-1",
        duration_ms=123.45,
    )

    json_str = ctx.to_json()

    # Should be valid JSON
    parsed = json.loads(json_str)
    assert parsed["level"] == "WARNING"
    assert parsed["message"] == "Warning message"
    assert parsed["session_id"] == "sess-1"
    assert parsed["duration_ms"] == 123.45


def test_log_context_excludes_none():
    """Test LogContext excludes None values from JSON."""
    ctx = LogContext(
        message="Test",
        duration_ms=None,
    )

    json_str = ctx.to_json()
    parsed = json.loads(json_str)

    assert "duration_ms" not in parsed


def test_structured_formatter_formats_record():
    """Test StructuredFormatter creates JSON logs."""
    record = logging.LogRecord(
        name="t4dm.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatter = StructuredFormatter()
    result = formatter.format(record)

    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed["message"] == "Test message"
    assert parsed["level"] == "INFO"


def test_structured_formatter_with_exception():
    """Test StructuredFormatter includes exception info."""
    try:
        raise ValueError("Test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="t4dm.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        formatter = StructuredFormatter()
        result = formatter.format(record)

        parsed = json.loads(result)
        # Exception is flattened into main dict by LogContext.to_json()
        assert "exception" in parsed


def test_context_adapter_adds_context():
    """Test ContextAdapter adds context to logs."""
    logger_instance = logging.getLogger("t4dm.test")
    adapter = ContextAdapter(logger_instance, {})

    set_context("sess-123", "op-456")

    msg, kwargs = adapter.process("Test message", {})

    assert "extra" in kwargs
    extra = kwargs["extra"]
    assert extra["session_id"] == "sess-123"
    assert extra["operation_id"] == "op-456"


def test_configure_logging_json():
    """Test configure_logging with JSON output."""
    configure_logging(level="INFO", json_output=True)

    # Handlers are added to root logger, not "ww" logger
    root_logger = logging.getLogger()
    handler = root_logger.handlers[0] if root_logger.handlers else None

    assert handler is not None
    assert isinstance(handler.formatter, StructuredFormatter)


def test_configure_logging_plain():
    """Test configure_logging with plain text output."""
    configure_logging(level="DEBUG", json_output=False)

    # Handlers are added to root logger, not "ww" logger
    root_logger = logging.getLogger()
    handler = root_logger.handlers[0] if root_logger.handlers else None

    assert handler is not None
    assert not isinstance(handler.formatter, StructuredFormatter)


def test_configure_logging_with_file():
    """Test configure_logging with file output."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")

        configure_logging(level="INFO", log_file=log_file)

        # Log a message
        logger_instance = logging.getLogger("t4dm")
        logger_instance.info("Test message")

        # File should exist and contain logs
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert len(content) > 0


def test_get_logger():
    """Test get_logger returns configured logger."""
    logger_instance = get_logger("t4dm.test.module")

    assert logger_instance.name == "t4dm.test.module"


def test_set_context():
    """Test set_context stores values in context vars."""
    set_context("session-abc", "operation-xyz")

    from t4dm.observability.logging import _session_id, _operation_id

    assert _session_id.get() == "session-abc"
    assert _operation_id.get() == "operation-xyz"


def test_clear_context():
    """Test clear_context resets context vars."""
    set_context("session-123", "operation-456")
    clear_context()

    from t4dm.observability.logging import _session_id, _operation_id

    assert _session_id.get() == "unknown"
    assert _operation_id.get() == "unknown"


# =============================================================================
# OperationLogger Tests
# =============================================================================


@pytest.mark.asyncio
async def test_operation_logger_success():
    """Test OperationLogger logs successful operations."""
    async with OperationLogger("test_operation", session_id="sess-1") as log:
        await asyncio.sleep(0.01)
        log.set_result(item_count=5)

    # Logger should have completed
    assert log.operation == "test_operation"
    assert log.result["item_count"] == 5


@pytest.mark.asyncio
async def test_operation_logger_exception():
    """Test OperationLogger logs operation exceptions."""
    try:
        async with OperationLogger("test_operation", session_id="sess-1") as log:
            raise ValueError("Test error")
    except ValueError:
        pass

    # Should have logged error


@pytest.mark.asyncio
async def test_operation_logger_timing():
    """Test OperationLogger records operation timing."""
    async with OperationLogger("test_operation") as log:
        await asyncio.sleep(0.02)

    # Duration should be recorded
    start_time = log.start_time
    assert start_time > 0


@pytest.mark.asyncio
async def test_log_operation_decorator():
    """Test log_operation decorator."""
    @log_operation("test_operation", session_id="sess-1")
    async def test_function():
        await asyncio.sleep(0.01)
        return "success"

    result = await test_function()
    assert result == "success"


# =============================================================================
# Metrics Collection Tests
# =============================================================================


def test_operation_metrics_creation():
    """Test OperationMetrics initialization."""
    metrics = OperationMetrics()

    assert metrics.count == 0
    assert metrics.error_count == 0
    assert metrics.avg_duration_ms == 0.0
    assert metrics.success_rate == 1.0


def test_operation_metrics_record_success():
    """Test OperationMetrics records successful operation."""
    metrics = OperationMetrics()

    metrics.record(duration_ms=100.0, error=False)
    metrics.record(duration_ms=150.0, error=False)

    assert metrics.count == 2
    assert metrics.error_count == 0
    assert metrics.success_rate == 1.0
    assert metrics.avg_duration_ms == 125.0


def test_operation_metrics_record_error():
    """Test OperationMetrics records failed operation."""
    metrics = OperationMetrics()

    metrics.record(duration_ms=100.0, error=False)
    metrics.record(duration_ms=50.0, error=True)

    assert metrics.count == 2
    assert metrics.error_count == 1
    assert metrics.success_rate == 0.5


def test_operation_metrics_min_max():
    """Test OperationMetrics tracks min/max duration."""
    metrics = OperationMetrics()

    metrics.record(duration_ms=100.0)
    metrics.record(duration_ms=50.0)
    metrics.record(duration_ms=200.0)

    assert metrics.min_duration_ms == 50.0
    assert metrics.max_duration_ms == 200.0


def test_operation_metrics_to_dict():
    """Test OperationMetrics converts to dictionary."""
    metrics = OperationMetrics()
    metrics.record(duration_ms=100.0)

    result = metrics.to_dict()

    assert result["count"] == 1
    assert result["error_count"] == 0
    assert "success_rate" in result
    assert "avg_duration_ms" in result


def test_metrics_collector_creation():
    """Test MetricsCollector initialization."""
    collector = MetricsCollector()

    assert collector._start_time is not None
    assert len(collector._operations) == 0
    assert len(collector._gauges) == 0


def test_metrics_collector_record_operation():
    """Test MetricsCollector records operations."""
    collector = MetricsCollector()

    collector.record_operation("api.request", duration_ms=50.0)
    collector.record_operation("api.request", duration_ms=75.0)

    assert "api.request" in collector._operations
    assert collector._operations["api.request"].count == 2


def test_metrics_collector_record_with_tags():
    """Test MetricsCollector records operations with tags."""
    collector = MetricsCollector()

    collector.record_operation("db.query", duration_ms=100.0, type="select")
    collector.record_operation("db.query", duration_ms=150.0, type="insert")

    # Should create separate entries per tag combination
    assert collector._operations["db.query[type=insert]"].count == 1
    assert collector._operations["db.query[type=select]"].count == 1


def test_metrics_collector_set_gauge():
    """Test MetricsCollector sets gauge values."""
    collector = MetricsCollector()

    collector.set_gauge("memory_usage_mb", 512.5)
    collector.set_gauge("queue_length", 25.0)

    assert collector._gauges["memory_usage_mb"] == 512.5
    assert collector._gauges["queue_length"] == 25.0


def test_metrics_collector_increment_counter():
    """Test MetricsCollector increments counters."""
    collector = MetricsCollector()

    collector.increment_counter("http_requests")
    collector.increment_counter("http_requests", 5)

    assert collector._operations["http_requests"].count == 6


def test_metrics_collector_get_metrics():
    """Test MetricsCollector returns all metrics."""
    collector = MetricsCollector()

    collector.record_operation("op1", duration_ms=100.0)
    collector.set_gauge("gauge1", 50.0)

    result = collector.get_metrics()

    assert "uptime_seconds" in result
    assert "operations" in result
    assert "gauges" in result


def test_metrics_collector_get_operation_metrics():
    """Test MetricsCollector retrieves specific operation metrics."""
    collector = MetricsCollector()

    collector.record_operation("test_op", duration_ms=75.0)

    result = collector.get_operation_metrics("test_op")

    assert result is not None
    assert result["count"] == 1
    assert result["avg_duration_ms"] == 75.0


def test_metrics_collector_get_operation_metrics_missing():
    """Test MetricsCollector returns None for missing operation."""
    collector = MetricsCollector()

    result = collector.get_operation_metrics("nonexistent")

    assert result is None


def test_metrics_collector_get_summary():
    """Test MetricsCollector returns summary statistics."""
    collector = MetricsCollector()

    collector.record_operation("op1", duration_ms=100.0)
    collector.record_operation("op2", duration_ms=50.0, error=True)
    collector.set_gauge("metric1", 100.0)

    summary = collector.get_summary()

    assert summary["total_operations"] == 2
    assert summary["total_errors"] == 1
    assert "overall_success_rate" in summary
    assert "slowest_operations" in summary
    assert "error_prone_operations" in summary


def test_metrics_collector_reset():
    """Test MetricsCollector can reset all metrics."""
    collector = MetricsCollector()

    collector.record_operation("op1", duration_ms=100.0)
    collector.set_gauge("gauge1", 50.0)

    collector.reset()

    assert len(collector._operations) == 0
    assert len(collector._gauges) == 0


def test_get_metrics_singleton():
    """Test get_metrics returns singleton instance."""
    metrics1 = get_metrics()
    metrics2 = get_metrics()

    assert metrics1 is metrics2


@pytest.mark.asyncio
async def test_timed_operation_decorator():
    """Test timed_operation decorator."""
    @timed_operation("test_op", category="test")
    async def test_function():
        await asyncio.sleep(0.01)
        return "result"

    result = await test_function()
    assert result == "result"

    # Check metrics were recorded
    metrics = get_metrics()
    assert "test_op[category=test]" in metrics._operations


@pytest.mark.asyncio
async def test_timed_operation_records_error():
    """Test timed_operation decorator records errors."""
    @timed_operation("failing_op")
    async def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        await failing_function()

    # Error should be recorded
    metrics = get_metrics()
    op_metrics = metrics.get_operation_metrics("failing_op")
    assert op_metrics["error_count"] == 1


@pytest.mark.asyncio
async def test_count_operation_decorator():
    """Test count_operation decorator."""
    # Reset metrics first
    metrics = get_metrics()
    metrics.reset()

    @count_operation("counted_op")
    async def test_function():
        return "result"

    await test_function()
    await test_function()

    op_metrics = metrics.get_operation_metrics("counted_op")
    assert op_metrics["count"] == 2


def test_timer_context_manager():
    """Test Timer context manager."""
    metrics = get_metrics()
    metrics.reset()

    with Timer("test_timer") as timer:
        time.sleep(0.01)

    assert timer.duration_ms > 0
    op_metrics = metrics.get_operation_metrics("test_timer")
    assert op_metrics is not None


def test_timer_context_manager_exception():
    """Test Timer records errors."""
    metrics = get_metrics()
    metrics.reset()

    try:
        with Timer("failing_timer"):
            raise ValueError("Test error")
    except ValueError:
        pass

    op_metrics = metrics.get_operation_metrics("failing_timer")
    assert op_metrics["error_count"] == 1


@pytest.mark.asyncio
async def test_async_timer_context_manager():
    """Test AsyncTimer async context manager."""
    metrics = get_metrics()
    metrics.reset()

    async with AsyncTimer("async_timer") as timer:
        await asyncio.sleep(0.01)

    assert timer.duration_ms > 0
    op_metrics = metrics.get_operation_metrics("async_timer")
    assert op_metrics is not None


@pytest.mark.asyncio
async def test_async_timer_exception():
    """Test AsyncTimer records errors."""
    metrics = get_metrics()
    metrics.reset()

    try:
        async with AsyncTimer("failing_async_timer"):
            raise ValueError("Test error")
    except ValueError:
        pass

    op_metrics = metrics.get_operation_metrics("failing_async_timer")
    assert op_metrics["error_count"] == 1


# =============================================================================
# Health Check Tests
# =============================================================================


def test_health_status_enum():
    """Test HealthStatus enum values."""
    assert HealthStatus.HEALTHY.value == "healthy"
    assert HealthStatus.DEGRADED.value == "degraded"
    assert HealthStatus.UNHEALTHY.value == "unhealthy"
    assert HealthStatus.UNKNOWN.value == "unknown"


def test_component_health_creation():
    """Test ComponentHealth creation."""
    health = ComponentHealth(
        name="test_component",
        status=HealthStatus.HEALTHY,
        message="All good",
        latency_ms=50.5,
    )

    assert health.name == "test_component"
    assert health.status == HealthStatus.HEALTHY
    assert health.latency_ms == 50.5


def test_component_health_to_dict():
    """Test ComponentHealth converts to dictionary."""
    health = ComponentHealth(
        name="qdrant",
        status=HealthStatus.HEALTHY,
        message="Connected",
        latency_ms=25.0,
        details={"points_count": 1000},
    )

    result = health.to_dict()

    assert result["name"] == "qdrant"
    assert result["status"] == "healthy"
    assert result["latency_ms"] == 25.0
    assert result["details"]["points_count"] == 1000


def test_system_health_to_dict():
    """Test SystemHealth converts to dictionary."""
    components = [
        ComponentHealth(
            name="qdrant",
            status=HealthStatus.HEALTHY,
        ),
        ComponentHealth(
            name="neo4j",
            status=HealthStatus.HEALTHY,
        ),
    ]

    system_health = SystemHealth(
        status=HealthStatus.HEALTHY,
        components=components,
        uptime_seconds=1000.0,
    )

    result = system_health.to_dict()

    assert result["status"] == "healthy"
    assert len(result["components"]) == 2
    assert result["uptime_seconds"] == 1000.0


def test_health_checker_creation():
    """Test HealthChecker initialization."""
    checker = HealthChecker(timeout=10.0)

    assert checker.timeout == 10.0


@pytest.mark.asyncio
async def test_health_checker_check_liveness():
    """Test check_liveness always returns True."""
    checker = HealthChecker()

    result = await checker.check_liveness()

    assert result is True


@pytest.mark.asyncio
async def test_health_checker_check_readiness():
    """Test check_readiness depends on component health."""
    checker = HealthChecker()

    with patch.object(checker, "check_all") as mock_check:
        # Mock healthy response
        mock_check.return_value = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=[],
        )

        result = await checker.check_readiness()
        assert result is True

    with patch.object(checker, "check_all") as mock_check:
        # Mock unhealthy response
        mock_check.return_value = SystemHealth(
            status=HealthStatus.UNHEALTHY,
            components=[],
        )

        result = await checker.check_readiness()
        assert result is False


@pytest.mark.skip(reason="Legacy test for old Qdrant architecture - T4DX uses embedded engine")
@pytest.mark.asyncio
async def test_health_checker_check_qdrant():
    """Test check_qdrant health check."""
    checker = HealthChecker()

    # Patch at source module since import happens inside the method
    with patch("t4dm.storage.qdrant_store.get_vector_store") as mock_get_store:
        mock_store = AsyncMock()
        mock_store.episodes_collection = "episodes"
        mock_store.count = AsyncMock(return_value=100)
        mock_get_store.return_value = mock_store

        result = await checker.check_qdrant()

        assert result.name == "qdrant"
        assert result.status == HealthStatus.HEALTHY


@pytest.mark.skip(reason="Legacy test for old Qdrant architecture - T4DX uses embedded engine")
@pytest.mark.asyncio
async def test_health_checker_check_qdrant_timeout():
    """Test check_qdrant timeout handling."""
    checker = HealthChecker(timeout=0.001)

    # Patch at source module since import happens inside the method
    with patch("t4dm.storage.qdrant_store.get_vector_store") as mock_get_store:
        mock_store = AsyncMock()
        mock_store.episodes_collection = "episodes"
        mock_store.count = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        mock_get_store.return_value = mock_store

        result = await checker.check_qdrant()

        assert result.name == "qdrant"
        assert result.status == HealthStatus.UNHEALTHY


@pytest.mark.skip(reason="Legacy test for old Neo4j architecture - T4DX uses embedded engine")
@pytest.mark.asyncio
async def test_health_checker_check_neo4j():
    """Test check_neo4j health check."""
    checker = HealthChecker()

    # Patch at source module since import happens inside the method
    with patch("t4dm.storage.neo4j_store.get_graph_store") as mock_get_store:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[{"n": 1}])
        mock_get_store.return_value = mock_store

        result = await checker.check_neo4j()

        assert result.name == "neo4j"
        assert result.status == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_health_checker_check_embedding():
    """Test check_embedding health check."""
    checker = HealthChecker()

    # Patch at source module since import happens inside the method
    with patch("t4dm.embedding.bge_m3.get_embedding_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.embed_query = AsyncMock(
            return_value=[0.1] * 1024
        )
        mock_get_provider.return_value = mock_provider

        result = await checker.check_embedding()

        assert result.name == "embedding"
        assert result.status == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_health_checker_check_metrics():
    """Test check_metrics health check."""
    checker = HealthChecker()

    result = await checker.check_metrics()

    assert result.name == "metrics"
    assert result.status == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_health_checker_check_all():
    """Test check_all runs all health checks."""
    checker = HealthChecker()

    with patch.object(checker, "check_t4dx") as mock_t4dx, \
         patch.object(checker, "check_embedding") as mock_embedding, \
         patch.object(checker, "check_metrics") as mock_metrics:

        mock_t4dx.return_value = ComponentHealth(
            name="t4dx",
            status=HealthStatus.HEALTHY,
        )
        mock_embedding.return_value = ComponentHealth(
            name="embedding",
            status=HealthStatus.HEALTHY,
        )
        mock_metrics.return_value = ComponentHealth(
            name="metrics",
            status=HealthStatus.HEALTHY,
        )

        result = await checker.check_all()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 3


def test_get_health_checker_singleton():
    """Test get_health_checker returns singleton."""
    checker1 = get_health_checker()
    checker2 = get_health_checker()

    assert checker1 is checker2
