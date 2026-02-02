"""
T4DM REST API Server.

FastAPI application with lifespan management for the tripartite memory system.

Features:
- REST API for memory operations
- WebSocket for real-time updates
- Persistence layer for crash recovery
- OpenTelemetry tracing
"""

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from t4dm.core.config import get_settings
from t4dm.core.services import cleanup_services
from t4dm.observability.tracing import init_tracing, shutdown_tracing

logger = logging.getLogger(__name__)

# P10.3: Graceful shutdown tracking
_shutdown_event = asyncio.Event()
_active_requests = 0
_active_requests_lock = asyncio.Lock()

# Global persistence manager reference
_persistence_manager = None
_health_metrics_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup initialization and shutdown cleanup.
    Includes persistence layer initialization for crash recovery.
    """
    global _persistence_manager, _health_metrics_task
    settings = get_settings()

    # Startup
    logger.info("Starting T4DM REST API Server")

    # Initialize tracing
    if settings.otel_enabled:
        init_tracing(
            service_name=settings.otel_service_name,
            otlp_endpoint=settings.otel_endpoint,
            console_export=settings.otel_console,
        )
        logger.info(f"OpenTelemetry tracing enabled: {settings.otel_endpoint}")

    # Initialize persistence layer
    try:
        from t4dm.persistence import PersistenceConfig, PersistenceManager

        data_dir = Path(os.environ.get("T4DM_DATA_DIR", "/var/lib/world-weaver"))
        data_dir.mkdir(parents=True, exist_ok=True)

        config = PersistenceConfig(
            data_directory=data_dir,
            checkpoint_interval_seconds=300.0,
            checkpoint_operation_threshold=1000,
        )
        _persistence_manager = PersistenceManager(config)

        result = await _persistence_manager.start()
        if result.success:
            logger.info(f"Persistence initialized: {result.mode.name}, LSN={result.checkpoint_lsn}")
        else:
            logger.warning(f"Persistence initialization had issues: {result.errors}")

    except ImportError:
        logger.warning("Persistence module not available, running without durability")
    except Exception as e:
        logger.error(f"Failed to initialize persistence: {e}")

    # Start health metrics broadcaster for WebSocket
    try:
        from t4dm.api.websocket import health_metrics_broadcaster
        _health_metrics_task = asyncio.create_task(health_metrics_broadcaster(interval=5.0))
        logger.info("Health metrics broadcaster started")
    except ImportError:
        logger.debug("WebSocket module not available")
    except Exception as e:
        logger.warning(f"Failed to start health broadcaster: {e}")

    # P10.3: Setup graceful shutdown signal handler
    loop = asyncio.get_event_loop()

    def handle_sigterm():
        """Handle SIGTERM for graceful shutdown (K8s pod eviction)."""
        logger.info("SIGTERM received, initiating graceful shutdown")
        _shutdown_event.set()

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_sigterm)

    logger.info("Graceful shutdown handlers registered")

    yield

    # Shutdown
    logger.info("Shutting down T4DM REST API Server")

    # P10.3: Wait for in-flight requests to complete (with timeout)
    drain_timeout = 30.0  # seconds
    drain_start = asyncio.get_event_loop().time()
    while _active_requests > 0:
        elapsed = asyncio.get_event_loop().time() - drain_start
        if elapsed > drain_timeout:
            logger.warning(f"Connection drain timeout: {_active_requests} requests still active")
            break
        logger.info(f"Draining connections: {_active_requests} requests in flight")
        await asyncio.sleep(0.5)

    if _active_requests == 0:
        logger.info("All connections drained successfully")

    # Stop health metrics broadcaster
    if _health_metrics_task:
        _health_metrics_task.cancel()
        try:
            await _health_metrics_task
        except asyncio.CancelledError:
            pass

    # Shutdown persistence (creates final checkpoint)
    if _persistence_manager:
        await _persistence_manager.shutdown()
        logger.info("Persistence shutdown complete")

    await cleanup_services()
    shutdown_tracing()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="T4DM Memory API",
    description="""
REST API for T4DM's tripartite neural memory system.

## Memory Subsystems

- **Episodic Memory**: Time-sequenced experiences with FSRS decay
- **Semantic Memory**: Knowledge graph with ACT-R activation and Hebbian learning
- **Procedural Memory**: Skill patterns with execution tracking

## Authentication

Pass session ID via `X-Session-ID` header. Defaults to configured session if not provided.
    """,
    version="0.5.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# P2-SEC-M6: Security headers middleware for XSS/clickjacking protection
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        # XSS Protection
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # HSTS for production (only if request is HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # CSP for API (restrictive since we serve JSON, not HTML)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# API-HIGH-001/002 FIX: Configure CORS with validated settings-based origins
# Security notes:
# - allow_credentials=True requires explicit origins (validated in config)
# - Wildcards (*) rejected in production mode
# - HTTPS required for non-localhost origins in production
# - CORSMiddleware reflects request origin if in allowed list (not wildcard response)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=settings.cors_allowed_headers + ["X-Admin-Key", "X-API-Key"],  # P2-SEC-M4: Include API key header
)

# Add security headers after CORS (order matters - middleware applied in reverse order)
app.add_middleware(SecurityHeadersMiddleware)


# P3-SEC-L7: Request size limit middleware to prevent DoS via large payloads
# ATOM-P0-10: Enhanced to prevent chunked transfer encoding bypass
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent DoS attacks."""

    MAX_REQUEST_SIZE = 5 * 1024 * 1024  # 5MB default limit

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only check body size for methods that can have request bodies
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length:
            # Content-Length header present: check size upfront
            try:
                if int(content_length) > self.MAX_REQUEST_SIZE:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body too large. Maximum size is {self.MAX_REQUEST_SIZE // (1024*1024)}MB"},
                    )
            except ValueError:
                pass  # Invalid content-length header, let it proceed
        else:
            # No Content-Length: chunked transfer or no body
            # Wrap receive to count bytes as they arrive
            bytes_received = 0
            original_receive = request.receive

            async def counting_receive():
                nonlocal bytes_received
                message = await original_receive()
                if message.get("type") == "http.request":
                    body = message.get("body", b"")
                    bytes_received += len(body)
                    if bytes_received > self.MAX_REQUEST_SIZE:
                        # Raise error to abort request processing
                        from fastapi.responses import JSONResponse
                        raise ValueError(f"Request body exceeds {self.MAX_REQUEST_SIZE // (1024*1024)}MB")
                return message

            # Replace receive callable
            request._receive = counting_receive

        return await call_next(request)


app.add_middleware(RequestSizeLimitMiddleware)


# P10.3: Request tracking middleware for graceful shutdown
class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Track active requests for connection draining during shutdown."""

    async def dispatch(self, request: Request, call_next) -> Response:
        global _active_requests

        # Check if shutdown is in progress
        if _shutdown_event.is_set():
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is shutting down"},
                headers={"Connection": "close", "Retry-After": "30"},
            )

        # Track request
        async with _active_requests_lock:
            _active_requests += 1

        try:
            response = await call_next(request)
            return response
        finally:
            async with _active_requests_lock:
                _active_requests -= 1


app.add_middleware(RequestTrackingMiddleware)


# P2-SEC-M4: API key authentication middleware
class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Require API key for all endpoints if configured.

    Exemptions:
    - Health check endpoint (for load balancer probes)
    - Root endpoint (redirect to docs)
    - OpenAPI schema (required for docs UI)
    - OPTIONS requests (for CORS preflight)
    ATOM-P2-13: /metrics removed from exempt paths (now requires authentication)
    """

    EXEMPT_PATHS = {"/", "/api/v1/health", "/openapi.json", "/docs", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Check API key requirement
        settings = get_settings()
        api_key = getattr(settings, "api_key", None)
        api_key_required = getattr(settings, "api_key_required", False)
        env = getattr(settings, "environment", "development")

        # Auto-enable in production if key is configured
        if env == "production" and api_key:
            api_key_required = True

        if not api_key_required or not api_key:
            return await call_next(request)

        # Validate API key
        import secrets

        from fastapi.responses import JSONResponse

        x_api_key = request.headers.get("x-api-key")

        if not x_api_key:
            logger.warning(f"Missing API key for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-API-Key header"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if not secrets.compare_digest(x_api_key, api_key):
            logger.warning(f"Invalid API key for {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)


app.add_middleware(ApiKeyAuthMiddleware)


# P3B: Rate limiting middleware (Phase 3B production infrastructure)
from t4dm.api.middleware.rate_limit import RateLimitMiddleware

# Add rate limiting with token bucket algorithm
# Default: 100 requests/min per client, 200 burst capacity
# Clients isolated by IP or API key
app.add_middleware(RateLimitMiddleware, rate=100, burst=200)


# Import and include routers
from t4dm.api.routes import (
    compat_router,
    config_router,
    dream_router,
    entities_router,
    episodes_router,
    # Phase 11: Demo APIs
    explorer_router,
    learning_router,
    nt_router,
    persistence_router,
    skills_router,
    system_router,
    visualization_router,
    diagrams_router,
)

app.include_router(episodes_router, prefix="/api/v1/episodes", tags=["Episodic Memory"])
app.include_router(entities_router, prefix="/api/v1/entities", tags=["Semantic Memory"])
app.include_router(skills_router, prefix="/api/v1/skills", tags=["Procedural Memory"])
app.include_router(config_router, prefix="/api/v1/config", tags=["Configuration"])
app.include_router(system_router, prefix="/api/v1", tags=["System"])
app.include_router(visualization_router, prefix="/api/v1/viz", tags=["Visualization"])
app.include_router(persistence_router, tags=["Persistence"])
app.include_router(diagrams_router, prefix="/api/v1/diagrams", tags=["Diagrams"])

# Phase 11: Demo APIs for interactive visualization
app.include_router(explorer_router, prefix="/api/v1/demo/explorer", tags=["Demo: Memory Explorer"])
app.include_router(dream_router, prefix="/api/v1/demo/dream", tags=["Demo: Dream Viewer"])
app.include_router(nt_router, prefix="/api/v1/demo/nt", tags=["Demo: NT Dashboard"])
app.include_router(learning_router, prefix="/api/v1/demo/learning", tags=["Demo: Learning Trace"])

# Mem0-compatible API
app.include_router(compat_router, tags=["Mem0 Compatibility"])

# Include WebSocket routes
try:
    from t4dm.api.websocket import router as websocket_router
    app.include_router(websocket_router, tags=["WebSocket"])
    logger.info("WebSocket routes registered")
except ImportError:
    logger.debug("WebSocket module not available")

# Include visualization WebSocket route
try:
    from t4dm.api.routes.ws_viz import router as ws_viz_router
    app.include_router(ws_viz_router, tags=["WebSocket"])
    logger.info("Visualization WebSocket route registered")
except ImportError:
    logger.debug("Visualization WebSocket module not available")

# P10.2: Include Prometheus metrics endpoint
try:
    from t4dm.observability.prometheus import prometheus_router
    if prometheus_router is not None:
        app.include_router(prometheus_router, tags=["Observability"])
        logger.info("Prometheus metrics endpoint registered at /metrics")
except ImportError:
    logger.debug("Prometheus router not available")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return {"message": "T4DM API", "docs": "/docs", "health": "/api/v1/health"}


def main():
    """Run the REST API server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")

    uvicorn.run(
        "t4dm.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
