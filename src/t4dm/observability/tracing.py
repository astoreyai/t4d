"""
OpenTelemetry tracing configuration for World Weaver.

Supports:
- Secure TLS connections for production
- Configurable batch export parameters
- Header-based authentication
- Resource attributes
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: trace.Tracer | None = None
_tracer_provider: TracerProvider | None = None


def configure_tracing() -> TracerProvider | None:
    """
    Configure OpenTelemetry tracing.

    Returns:
        Configured TracerProvider or None if disabled
    """
    global _tracer, _tracer_provider

    from t4dm.core.config import get_settings
    settings = get_settings()

    if not settings.otel_enabled:
        logger.info("OpenTelemetry tracing disabled")
        return None

    if _tracer_provider is not None:
        logger.debug("TracerProvider already configured")
        return _tracer_provider

    # Build resource attributes
    resource = Resource.create({
        SERVICE_NAME: settings.otel_service_name,
        SERVICE_VERSION: "0.1.0",
        "deployment.environment": settings.environment,
    })

    # Configure exporter with security
    exporter_kwargs = {
        "endpoint": settings.otel_endpoint,
    }

    # TLS configuration
    if not settings.otel_insecure:
        if settings.otel_cert_file:
            # Load custom certificate
            try:
                with open(settings.otel_cert_file, "rb") as f:
                    cert_data = f.read()
                from grpc import ssl_channel_credentials
                credentials = ssl_channel_credentials(root_certificates=cert_data)
                exporter_kwargs["credentials"] = credentials
                logger.info(f"Using custom TLS certificate from {settings.otel_cert_file}")
            except Exception as e:
                logger.error(f"Failed to load TLS certificate: {e}")
                raise
        else:
            # Use system certificates
            exporter_kwargs["insecure"] = False
            logger.info("Using system TLS certificates for OTLP")
    else:
        exporter_kwargs["insecure"] = True
        logger.warning("OTLP exporter using insecure connection (not for production)")

    # Add authentication headers if provided
    if settings.otel_headers:
        exporter_kwargs["headers"] = tuple(settings.otel_headers.items())
        logger.debug(f"OTLP exporter configured with {len(settings.otel_headers)} headers")

    # Create exporter
    try:
        exporter = OTLPSpanExporter(**exporter_kwargs)
    except Exception as e:
        logger.error(f"Failed to create OTLP exporter: {e}")
        raise

    # Configure batch processor
    processor = BatchSpanProcessor(
        exporter,
        max_export_batch_size=settings.otel_max_export_batch_size,
        schedule_delay_millis=settings.otel_batch_delay_ms,
    )

    # Create and configure provider
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(processor)

    # Add console exporter for debugging
    if settings.otel_console:
        logger.info("Enabling console span export for debugging")
        console_exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set as global provider
    trace.set_tracer_provider(_tracer_provider)

    # Get tracer
    _tracer = trace.get_tracer("world-weaver", "0.1.0")

    logger.info(
        f"OpenTelemetry tracing configured: "
        f"endpoint={settings.otel_endpoint}, "
        f"secure={not settings.otel_insecure}, "
        f"batch_size={settings.otel_max_export_batch_size}"
    )

    return _tracer_provider


def init_tracing(
    service_name: str = "world-weaver",
    otlp_endpoint: str | None = None,
    console_export: bool = False,
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing (legacy compatibility wrapper).

    Args:
        service_name: Service name for traces (ignored, use config)
        otlp_endpoint: OTLP gRPC endpoint (ignored, use config)
        console_export: Export spans to console for debugging (ignored, use config)

    Returns:
        Configured tracer instance
    """
    global _tracer

    # Configure using settings
    provider = configure_tracing()

    if provider is None:
        # Return no-op tracer
        _tracer = trace.get_tracer("world-weaver")
        return _tracer

    return _tracer


def get_tracer(name: str = "world-weaver") -> trace.Tracer:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually module name)

    Returns:
        Tracer instance
    """
    global _tracer

    if _tracer is None:
        # Auto-initialize with defaults
        configure_tracing()

    provider = trace.get_tracer_provider()
    return provider.get_tracer(name, "0.1.0")


def shutdown_tracing() -> None:
    """Shutdown tracing and flush pending spans."""
    global _tracer_provider, _tracer

    if _tracer_provider is not None:
        logger.info("Shutting down OpenTelemetry tracer")
        _tracer_provider.shutdown()
        _tracer_provider = None
        _tracer = None


def traced(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict | None = None,
):
    """
    Decorator to trace async functions.

    Args:
        name: Span name (defaults to module.function)
        kind: Span kind (SERVER, CLIENT, INTERNAL, etc.)
        attributes: Static attributes to add to span

    Returns:
        Decorated async function

    Example:
        @traced("create_episode", kind=SpanKind.SERVER)
        async def create_episode(content: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name, kind=kind) as span:
                # Add static attributes
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))

                # Extract common context from kwargs
                if "session_id" in kwargs:
                    span.set_attribute("t4dm.session_id", kwargs["session_id"])

                # Add function name for easy filtering
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def traced_sync(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict | None = None,
):
    """
    Decorator to trace synchronous functions.

    Args:
        name: Span name (defaults to module.function)
        kind: Span kind (SERVER, CLIENT, INTERNAL, etc.)
        attributes: Static attributes to add to span

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name, kind=kind) as span:
                # Add static attributes
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


@asynccontextmanager
async def trace_span(name: str, kind: SpanKind = SpanKind.INTERNAL, **attributes):
    """
    Context manager for tracing code blocks.

    Args:
        name: Span name
        kind: Span kind
        **attributes: Key-value pairs to add as span attributes

    Example:
        async with trace_span("batch_processing", batch_size=100):
            await process_batch()
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name, kind=kind) as span:
        # Add attributes
        for k, v in attributes.items():
            span.set_attribute(k, str(v))

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def add_span_attribute(key: str, value: Any) -> None:
    """
    Add attribute to current active span.

    Args:
        key: Attribute key
        value: Attribute value (will be stringified)

    Example:
        add_span_attribute("db.query", "SELECT * FROM episodes")
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: dict | None = None) -> None:
    """
    Add event to current active span.

    Args:
        name: Event name
        attributes: Optional event attributes

    Example:
        add_span_event("cache_miss", {"key": "episode_123"})
    """
    span = trace.get_current_span()
    if span.is_recording():
        attrs = {k: str(v) for k, v in (attributes or {}).items()}
        span.add_event(name, attributes=attrs)


def get_trace_context() -> dict:
    """
    Get current trace context for propagation.

    Returns:
        Dict with traceparent and tracestate for W3C context propagation
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return {}

    ctx = span.get_span_context()
    if not ctx.is_valid:
        return {}

    # W3C trace context format
    traceparent = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"

    result = {"traceparent": traceparent}

    if ctx.trace_state:
        result["tracestate"] = ctx.trace_state.to_header()

    return result
