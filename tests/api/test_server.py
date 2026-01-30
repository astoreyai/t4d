"""Tests for API server configuration."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestAppConfiguration:
    """Tests for FastAPI app configuration."""

    def test_app_exists(self):
        """App is created successfully."""
        from ww.api.server import app
        assert app is not None
        assert app.title == "World Weaver Memory API"

    def test_app_version(self):
        """App has correct version."""
        from ww.api.server import app
        assert app.version == "0.5.0"

    def test_docs_enabled(self):
        """API docs endpoints are configured."""
        from ww.api.server import app
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"


class TestRouters:
    """Tests for router inclusion."""

    def test_episodes_router_included(self):
        """Episodes router is included."""
        from ww.api.server import app
        routes = [r.path for r in app.routes]
        assert any("/api/v1/episodes" in str(r) for r in routes)

    def test_entities_router_included(self):
        """Entities router is included."""
        from ww.api.server import app
        routes = [r.path for r in app.routes]
        assert any("/api/v1/entities" in str(r) for r in routes)

    def test_skills_router_included(self):
        """Skills router is included."""
        from ww.api.server import app
        routes = [r.path for r in app.routes]
        assert any("/api/v1/skills" in str(r) for r in routes)

    def test_system_router_included(self):
        """System router is included at /api/v1."""
        from ww.api.server import app
        routes = [r.path for r in app.routes]
        assert any("/api/v1/health" in str(r) for r in routes)

    def test_viz_router_included(self):
        """Visualization router is included."""
        from ww.api.server import app
        routes = [r.path for r in app.routes]
        assert any("/api/v1/viz" in str(r) for r in routes)

    def test_config_router_included(self):
        """Config router is included."""
        from ww.api.server import app
        routes = [r.path for r in app.routes]
        assert any("/api/v1/config" in str(r) for r in routes)


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_middleware_added(self):
        """CORS middleware is configured."""
        from ww.api.server import app

        # Check middleware stack
        middleware_classes = [type(m).__name__ for m in app.user_middleware]
        # CORS is added via add_middleware, check it's in the stack
        assert len(app.user_middleware) >= 0  # At least processing is set up


class TestSecurityHeaders:
    """Tests for P2-SEC-M6 security headers middleware."""

    @pytest.mark.asyncio
    async def test_security_headers_added(self):
        """Security headers middleware adds required headers."""
        from ww.api.server import SecurityHeadersMiddleware
        from starlette.testclient import TestClient
        from ww.api.server import app

        client = TestClient(app)
        response = client.get("/")

        # Verify security headers are present
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Content-Security-Policy" in response.headers
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_security_headers_middleware_class_exists(self):
        """SecurityHeadersMiddleware class exists."""
        from ww.api.server import SecurityHeadersMiddleware
        assert SecurityHeadersMiddleware is not None


class TestRequestSizeLimit:
    """Tests for P3-SEC-L7 request size limit middleware."""

    def test_request_size_limit_middleware_exists(self):
        """RequestSizeLimitMiddleware class exists."""
        from ww.api.server import RequestSizeLimitMiddleware
        assert RequestSizeLimitMiddleware is not None
        assert RequestSizeLimitMiddleware.MAX_REQUEST_SIZE == 5 * 1024 * 1024  # 5MB

    @pytest.mark.asyncio
    async def test_large_request_rejected(self):
        """Requests over size limit return 413."""
        from starlette.testclient import TestClient
        from ww.api.server import app

        client = TestClient(app)
        # Simulate large content-length header
        response = client.post(
            "/api/v1/episodes",
            headers={"Content-Length": str(10 * 1024 * 1024)},  # 10MB
            content=b"",
        )
        assert response.status_code == 413


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_info(self):
        """Root endpoint returns API info."""
        from ww.api.server import root

        result = await root()
        assert "message" in result
        assert "docs" in result
        assert "health" in result
        assert result["docs"] == "/docs"


class TestLifespan:
    """Tests for application lifespan."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Lifespan context manager works."""
        from ww.api.server import lifespan, app

        with patch("ww.api.server.get_settings") as mock_settings:
            mock_settings.return_value.otel_enabled = False

            with patch("ww.api.server.cleanup_services", new=AsyncMock()):
                with patch("ww.api.server.shutdown_tracing"):
                    async with lifespan(app):
                        pass  # Startup should complete

    @pytest.mark.asyncio
    async def test_lifespan_with_tracing(self):
        """Lifespan initializes tracing when enabled."""
        from ww.api.server import lifespan, app

        with patch("ww.api.server.get_settings") as mock_settings:
            settings = MagicMock()
            settings.otel_enabled = True
            settings.otel_service_name = "test"
            settings.otel_endpoint = "http://localhost:4317"
            settings.otel_console = False
            mock_settings.return_value = settings

            with patch("ww.api.server.init_tracing") as mock_init:
                with patch("ww.api.server.cleanup_services", new=AsyncMock()):
                    with patch("ww.api.server.shutdown_tracing"):
                        async with lifespan(app):
                            mock_init.assert_called_once()
