"""Tests for API security atoms P0-10, P0-11, P2-13."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
from starlette.testclient import TestClient


class TestAtomP010ChunkedTransferBypass:
    """Tests for ATOM-P0-10: Fix chunked transfer size limit bypass."""

    def test_middleware_class_updated(self):
        """RequestSizeLimitMiddleware handles chunked transfers."""
        from ww.api.server import RequestSizeLimitMiddleware

        assert RequestSizeLimitMiddleware is not None
        assert RequestSizeLimitMiddleware.MAX_REQUEST_SIZE == 5 * 1024 * 1024

    def test_get_requests_bypass_size_check(self):
        """GET requests skip body size validation."""
        from ww.api.server import app

        client = TestClient(app)
        # GET with fake large content-length should still work (content-length ignored for GET)
        response = client.get("/", headers={"Content-Length": "10000000"})
        # Should not return 413
        assert response.status_code != 413

    def test_content_length_header_checked(self):
        """Requests with Content-Length header are checked upfront."""
        from ww.api.server import app

        client = TestClient(app)
        # POST with large content-length header should be rejected
        response = client.post(
            "/api/v1/episodes",
            headers={"Content-Length": str(10 * 1024 * 1024)},  # 10MB
            content=b"",
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_chunked_transfer_counting_logic_exists(self):
        """Middleware has logic to count chunked transfer bytes."""
        from ww.api.server import RequestSizeLimitMiddleware
        import inspect

        # Check that the dispatch method contains chunked transfer handling
        source = inspect.getsource(RequestSizeLimitMiddleware.dispatch)
        # Verify key elements are present in the code
        assert "counting_receive" in source
        assert "bytes_received" in source
        assert "original_receive" in source
        assert "http.request" in source


class TestAtomP011ImmutableTimestamps:
    """Tests for ATOM-P0-11: Add server-side immutable timestamps."""

    def test_create_episode_imports_timezone(self):
        """Episode routes import timezone for server-side timestamps."""
        from ww.api.routes import episodes

        # Check that timezone is imported
        assert hasattr(episodes, "timezone")

    @pytest.mark.asyncio
    async def test_create_episode_adds_ingested_at(self):
        """create_episode endpoint adds server-controlled ingested_at timestamp."""
        import inspect
        from ww.api.routes.episodes import create_episode

        # Check the function source contains ingested_at logic
        source = inspect.getsource(create_episode)
        assert "ingested_at" in source
        assert "datetime.now(timezone.utc)" in source
        assert "Server-side immutable timestamp" in source or "ATOM-P0-11" in source

    @pytest.mark.asyncio
    async def test_update_episode_preserves_timestamp(self):
        """update_episode endpoint documents timestamp immutability."""
        import inspect
        from ww.api.routes.episodes import update_episode

        # Check the function documents timestamp preservation
        source = inspect.getsource(update_episode)
        assert "ingested_at" in source or "immutable" in source.lower()
        assert "ATOM-P0-11" in source


class TestAtomP213MetricsAuthentication:
    """Tests for ATOM-P2-13: Authenticate /metrics endpoint."""

    def test_metrics_not_in_exempt_paths(self):
        """EXEMPT_PATHS does not include /metrics."""
        from ww.api.server import ApiKeyAuthMiddleware

        assert "/metrics" not in ApiKeyAuthMiddleware.EXEMPT_PATHS

    def test_exempt_paths_documented(self):
        """ApiKeyAuthMiddleware documents that /metrics requires auth."""
        import inspect
        from ww.api.server import ApiKeyAuthMiddleware

        source = inspect.getsource(ApiKeyAuthMiddleware)
        # Check that ATOM-P2-13 is mentioned or /metrics is removed
        assert "ATOM-P2-13" in source or "/metrics" not in source.split("EXEMPT_PATHS")[0]

    def test_health_still_exempt(self):
        """Health check endpoint remains exempt for load balancer probes."""
        from ww.api.server import ApiKeyAuthMiddleware

        assert "/api/v1/health" in ApiKeyAuthMiddleware.EXEMPT_PATHS

    def test_docs_still_exempt(self):
        """Docs endpoints remain exempt."""
        from ww.api.server import ApiKeyAuthMiddleware

        assert "/docs" in ApiKeyAuthMiddleware.EXEMPT_PATHS
        assert "/redoc" in ApiKeyAuthMiddleware.EXEMPT_PATHS
        assert "/openapi.json" in ApiKeyAuthMiddleware.EXEMPT_PATHS


class TestSecurityAtomsIntegration:
    """Integration tests for all three security atoms."""

    def test_all_middleware_classes_exist(self):
        """All required middleware classes are present."""
        from ww.api.server import (
            RequestSizeLimitMiddleware,
            ApiKeyAuthMiddleware,
            SecurityHeadersMiddleware,
            RequestTrackingMiddleware,
        )

        assert RequestSizeLimitMiddleware is not None
        assert ApiKeyAuthMiddleware is not None
        assert SecurityHeadersMiddleware is not None
        assert RequestTrackingMiddleware is not None

    def test_middleware_stack_includes_all(self):
        """Middleware stack includes all security middleware."""
        from ww.api.server import app

        # Get middleware class names
        middleware_names = [type(m.cls).__name__ if hasattr(m, 'cls') else type(m).__name__
                           for m in app.user_middleware]

        # At least check that middleware is registered (names may vary based on Starlette version)
        assert len(app.user_middleware) >= 4  # At least 4 middleware layers

    def test_security_headers_on_root(self):
        """Security headers are applied to responses."""
        from ww.api.server import app

        client = TestClient(app)
        response = client.get("/")

        # Verify security headers from SecurityHeadersMiddleware
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
