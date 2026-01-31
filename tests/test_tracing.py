"""
Tests for OpenTelemetry tracing integration.

Verifies that tracing initializes correctly and decorators work.
"""

import pytest
from opentelemetry.trace import SpanKind, get_current_span

from t4dm.observability.tracing import (
    init_tracing,
    get_tracer,
    shutdown_tracing,
    traced,
    traced_sync,
    trace_span,
    add_span_attribute,
    add_span_event,
    get_trace_context,
)


class TestTracingInitialization:
    """Test tracing initialization and configuration."""

    def test_init_tracing_default(self):
        """Test default tracing initialization."""
        tracer = init_tracing(service_name="test-service")
        assert tracer is not None
        shutdown_tracing()

    def test_get_tracer(self):
        """Test getting tracer instance."""
        tracer = get_tracer()
        assert tracer is not None
        shutdown_tracing()

    def test_get_trace_context(self):
        """Test getting trace context."""
        tracer = get_tracer()
        with tracer.start_as_current_span("test"):
            ctx = get_trace_context()
            # Should have traceparent when span is recording
            assert isinstance(ctx, dict)
        shutdown_tracing()


class TestTracingDecorators:
    """Test tracing decorators."""

    @pytest.mark.asyncio
    async def test_traced_decorator(self):
        """Test traced decorator for async functions."""
        call_count = 0

        @traced("test.function", kind=SpanKind.INTERNAL)
        async def test_function(value: int):
            nonlocal call_count
            call_count += 1
            return value * 2

        result = await test_function(5)
        assert result == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_traced_decorator_with_error(self):
        """Test traced decorator records exceptions."""

        @traced("test.error_function")
        async def error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await error_function()

    def test_traced_sync_decorator(self):
        """Test traced_sync decorator for sync functions."""
        call_count = 0

        @traced_sync("test.sync_function")
        def sync_function(value: int):
            nonlocal call_count
            call_count += 1
            return value + 1

        result = sync_function(10)
        assert result == 11
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_trace_span_context_manager(self):
        """Test trace_span context manager."""
        init_tracing(console_export=True)
        tracer = get_tracer()

        async with trace_span("test.span", kind=SpanKind.INTERNAL, test_attr="value"):
            # Verify span is active
            span = get_current_span()
            assert span is not None
            assert span.is_recording()

        shutdown_tracing()


class TestTracingHelpers:
    """Test tracing helper functions."""

    @pytest.mark.asyncio
    async def test_add_span_attribute(self):
        """Test adding attributes to active span."""
        init_tracing(console_export=True)
        tracer = get_tracer()

        with tracer.start_as_current_span("test") as span:
            add_span_attribute("test.key", "test.value")
            # Span should still be recording
            assert span.is_recording()

        shutdown_tracing()

    @pytest.mark.asyncio
    async def test_add_span_event(self):
        """Test adding events to active span."""
        init_tracing(console_export=True)
        tracer = get_tracer()

        with tracer.start_as_current_span("test") as span:
            add_span_event("test_event", {"key": "value"})
            # Span should still be recording
            assert span.is_recording()

        shutdown_tracing()


class TestTracingIntegration:
    """Integration tests for tracing through the stack."""

    @pytest.mark.asyncio
    async def test_nested_spans(self):
        """Test nested span creation."""

        @traced("outer.function")
        async def outer_function():
            async with trace_span("inner.span"):
                return "success"

        result = await outer_function()
        assert result == "success"

        shutdown_tracing()

    @pytest.mark.asyncio
    async def test_multiple_attributes(self):
        """Test adding multiple attributes."""
        init_tracing(console_export=True)
        tracer = get_tracer()

        with tracer.start_as_current_span("test") as span:
            for i in range(5):
                add_span_attribute(f"key_{i}", f"value_{i}")

            assert span.is_recording()

        shutdown_tracing()
