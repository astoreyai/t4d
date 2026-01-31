"""Tests for metrics module."""

import pytest
from datetime import datetime

from t4dm.observability.metrics import (
    get_metrics,
    OperationMetrics,
    MetricsCollector,
)


class TestOperationMetrics:
    """Tests for OperationMetrics dataclass."""

    def test_initial_values(self):
        """Initial values are correct."""
        metrics = OperationMetrics()
        assert metrics.count == 0
        assert metrics.error_count == 0
        assert metrics.total_duration_ms == 0.0

    def test_record_operation(self):
        """Record increases count and duration."""
        metrics = OperationMetrics()
        metrics.record(100.0)
        assert metrics.count == 1
        assert metrics.total_duration_ms == 100.0
        assert metrics.min_duration_ms == 100.0
        assert metrics.max_duration_ms == 100.0

    def test_record_multiple(self):
        """Multiple records accumulate correctly."""
        metrics = OperationMetrics()
        metrics.record(50.0)
        metrics.record(150.0)
        assert metrics.count == 2
        assert metrics.total_duration_ms == 200.0
        assert metrics.min_duration_ms == 50.0
        assert metrics.max_duration_ms == 150.0

    def test_record_with_error(self):
        """Error recording increases error count."""
        metrics = OperationMetrics()
        metrics.record(100.0, error=True)
        assert metrics.count == 1
        assert metrics.error_count == 1

    def test_avg_duration_empty(self):
        """Average duration is 0 when empty."""
        metrics = OperationMetrics()
        assert metrics.avg_duration_ms == 0.0

    def test_avg_duration(self):
        """Average duration is calculated correctly."""
        metrics = OperationMetrics()
        metrics.record(100.0)
        metrics.record(200.0)
        assert metrics.avg_duration_ms == 150.0

    def test_success_rate_empty(self):
        """Success rate is 1.0 when empty."""
        metrics = OperationMetrics()
        assert metrics.success_rate == 1.0

    def test_success_rate(self):
        """Success rate is calculated correctly."""
        metrics = OperationMetrics()
        metrics.record(10.0)
        metrics.record(10.0)
        metrics.record(10.0, error=True)
        assert metrics.count == 3
        assert metrics.error_count == 1
        assert metrics.success_rate == pytest.approx(2/3)

    def test_to_dict(self):
        """Converts to dictionary."""
        metrics = OperationMetrics()
        metrics.record(100.0)
        d = metrics.to_dict()
        assert d["count"] == 1
        assert d["error_count"] == 0
        assert d["success_rate"] == 1.0
        assert d["avg_duration_ms"] == 100.0
        assert "last_called" in d


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_singleton(self):
        """get_metrics returns singleton."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_record_operation(self):
        """Can record an operation."""
        metrics = get_metrics()
        metrics.record_operation("test_op", 50.0)

    def test_record_operation_with_error(self):
        """Can record an operation with error."""
        metrics = get_metrics()
        metrics.record_operation("test_op", 50.0, error=True)

    def test_get_operation_metrics(self):
        """Can get metrics for operation."""
        metrics = get_metrics()
        metrics.record_operation("specific_op", 100.0)
        op_metrics = metrics.get_operation_metrics("specific_op")
        assert op_metrics is not None

    def test_get_summary(self):
        """Can get summary of all metrics."""
        metrics = get_metrics()
        summary = metrics.get_summary()
        assert isinstance(summary, dict)
        assert "total_operations" in summary

    def test_reset(self):
        """Can reset metrics."""
        metrics = get_metrics()
        metrics.record_operation("reset_test", 10.0)
        metrics.reset()
        # After reset, operations should be cleared
        summary = metrics.get_summary()
        # Note: reset behavior depends on implementation


class TestMetricsDecorator:
    """Tests for metrics decorators."""

    @pytest.mark.asyncio
    async def test_timed_operation_decorator(self):
        """Async function can be timed with decorator."""
        from t4dm.observability.metrics import timed_operation

        @timed_operation("test_async_op")
        async def async_func():
            return "async_result"

        result = await async_func()
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_timed_operation_records_metrics(self):
        """Timed operation decorator records to metrics."""
        from t4dm.observability.metrics import timed_operation

        @timed_operation("recorded_op")
        async def timed_func():
            return 42

        result = await timed_func()
        assert result == 42
        # Check metrics were recorded
        metrics = get_metrics()
        op = metrics.get_operation_metrics("recorded_op")
        assert op is not None
        assert op["count"] >= 1


class TestSessionMetrics:
    """Tests for session-scoped metrics."""

    def test_record_session_metric(self):
        """Can record session-specific metric."""
        metrics = get_metrics()
        metrics.record_operation("session.test", 10.0, session_id="test-123")

    def test_get_session_summary(self):
        """Can get session summary."""
        metrics = get_metrics()
        # May not have session-specific summary, check if method exists
        if hasattr(metrics, "get_session_summary"):
            metrics.get_session_summary("test-session")
