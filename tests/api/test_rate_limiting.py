"""
Tests for API rate limiting middleware.

Phase 3B: Token bucket rate limiting with per-client isolation.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from ww.api.middleware.rate_limit import (
    RateLimitMiddleware,
    TokenBucket,
    TokenBucketRateLimiter,
    get_client_id,
    get_rate_limiter,
    rate_limit_middleware,
)


class TestTokenBucket:
    """Tests for TokenBucket algorithm."""

    @pytest.mark.asyncio
    async def test_initial_state(self):
        """Bucket starts with full capacity."""
        bucket = TokenBucket(rate=60, burst=100)
        tokens, capacity = bucket.get_state()
        assert tokens == 100.0
        assert capacity == 100

    @pytest.mark.asyncio
    async def test_consume_success(self):
        """Consuming tokens when available succeeds."""
        bucket = TokenBucket(rate=60, burst=100)
        allowed = await bucket.consume(1)
        assert allowed is True
        tokens, _ = bucket.get_state()
        assert tokens == 99.0

    @pytest.mark.asyncio
    async def test_consume_failure(self):
        """Consuming tokens when insufficient fails."""
        bucket = TokenBucket(rate=60, burst=1)
        # Consume first token
        await bucket.consume(1)
        # Second should fail (bucket empty)
        allowed = await bucket.consume(1)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Tokens refill over time at configured rate."""
        bucket = TokenBucket(rate=60, burst=100)  # 1 token per second

        # Consume all tokens
        for _ in range(100):
            await bucket.consume(1)

        tokens, _ = bucket.get_state()
        assert tokens < 1.0

        # Wait for refill (1 second = 1 token at rate=60/min)
        await asyncio.sleep(1.1)

        # Should have ~1 token now
        allowed = await bucket.consume(1)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_burst_capacity_limit(self):
        """Tokens don't exceed burst capacity."""
        bucket = TokenBucket(rate=60, burst=10)

        # Wait longer than needed to fill
        await asyncio.sleep(2.0)

        # Should still be capped at burst
        tokens, capacity = bucket.get_state()
        assert tokens <= capacity
        assert tokens <= 10.0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Bucket handles concurrent access correctly."""
        bucket = TokenBucket(rate=120, burst=100)

        async def consume_one():
            return await bucket.consume(1)

        # Try 50 concurrent requests
        results = await asyncio.gather(*[consume_one() for _ in range(50)])

        # All should succeed (100 tokens available)
        assert all(results)
        tokens, _ = bucket.get_state()
        # Allow for small timing variance in concurrent execution
        assert 49.5 <= tokens <= 50.5


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_creates_bucket_for_new_client(self):
        """Rate limiter creates bucket for new client."""
        limiter = TokenBucketRateLimiter(rate=60, burst=100)
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is True
        assert "client1" in limiter.buckets

    @pytest.mark.asyncio
    async def test_client_isolation(self):
        """Different clients have separate rate limits."""
        limiter = TokenBucketRateLimiter(rate=60, burst=2)

        # Client 1 uses both tokens
        await limiter.check_rate_limit("client1")
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is True

        # Client 1 should be blocked now
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is False

        # Client 2 should still be allowed (separate bucket)
        allowed, _ = await limiter.check_rate_limit("client2")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_returns_bucket_state(self):
        """Check returns current bucket state."""
        limiter = TokenBucketRateLimiter(rate=60, burst=100)

        allowed, (tokens, capacity) = await limiter.check_rate_limit("client1")
        assert allowed is True
        assert capacity == 100
        assert tokens == 99.0  # One token consumed

    @pytest.mark.asyncio
    async def test_get_bucket_state_nonexistent(self):
        """Getting state for nonexistent client returns None."""
        limiter = TokenBucketRateLimiter(rate=60, burst=100)
        state = limiter.get_bucket_state("nonexistent")
        assert state is None

    @pytest.mark.asyncio
    async def test_cleanup_inactive_buckets(self):
        """Cleanup removes inactive buckets."""
        limiter = TokenBucketRateLimiter(rate=60, burst=100)

        # Create bucket for client
        await limiter.check_rate_limit("client1")
        assert "client1" in limiter.buckets

        # Manually set last_update to old time
        limiter.buckets["client1"].last_update = time.monotonic() - 7200  # 2 hours ago

        # Cleanup with 1 hour threshold
        await limiter.cleanup_inactive_buckets(max_age_seconds=3600)

        # Bucket should be removed
        assert "client1" not in limiter.buckets


class TestGetClientId:
    """Tests for client ID extraction."""

    def test_uses_api_key_if_present(self):
        """Client ID from API key takes priority."""
        request = MagicMock(spec=Request)
        request.headers.get.side_effect = lambda k: {
            "x-api-key": "test-key-1234567890abcdef",
        }.get(k)

        client_id = get_client_id(request)
        assert client_id.startswith("key:")
        assert "test-key-123456" in client_id  # Only first 16 chars

    def test_uses_forwarded_for_if_no_key(self):
        """Client ID from X-Forwarded-For if no API key."""
        request = MagicMock(spec=Request)
        request.headers.get.side_effect = lambda k: {
            "x-forwarded-for": "192.168.1.100, 10.0.0.1",
        }.get(k)

        client_id = get_client_id(request)
        assert client_id == "ip:192.168.1.100"

    def test_uses_client_ip_as_fallback(self):
        """Client ID from direct IP as fallback."""
        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        request.client = MagicMock()
        request.client.host = "10.0.0.1"

        client_id = get_client_id(request)
        assert client_id == "ip:10.0.0.1"

    def test_handles_missing_client_info(self):
        """Returns 'unknown' if no client info available."""
        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        request.client = None

        client_id = get_client_id(request)
        assert client_id == "unknown"


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware integration."""

    def test_middleware_class_initializes(self):
        """Middleware class initializes correctly."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app, rate=100, burst=200)
        assert middleware.limiter is not None
        assert middleware.limiter.rate == 100
        assert middleware.limiter.burst == 200

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        """Requests under limit are allowed through."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app, rate=60, burst=100)

        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        request.url.path = "/test"

        call_next = AsyncMock(return_value=MagicMock(headers={}))

        response = await middleware.dispatch(request, call_next)

        # Should call next handler
        call_next.assert_called_once_with(request)
        # Should add rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        """Requests over limit return 429."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app, rate=60, burst=2)

        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        request.url.path = "/test"

        call_next = AsyncMock(return_value=MagicMock(headers={}))

        # First two requests succeed
        await middleware.dispatch(request, call_next)
        await middleware.dispatch(request, call_next)

        # Third request should fail
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 429
        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_429_response_format(self):
        """429 response has correct format."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app, rate=60, burst=1)

        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        request.url.path = "/test"

        call_next = AsyncMock()

        # Consume token
        await middleware.dispatch(request, call_next)

        # Next request should fail
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 429
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert response.headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in response.headers


class TestRateLimitMiddlewareFunction:
    """Tests for functional middleware."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        """Functional middleware allows requests under limit."""
        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        request.url.path = "/test"

        call_next = AsyncMock(return_value=MagicMock(headers={}))

        # Reset global limiter to known state
        with patch("ww.api.middleware.rate_limit._rate_limiter", None):
            response = await rate_limit_middleware(request, call_next)

        call_next.assert_called_once_with(request)
        assert "X-RateLimit-Limit" in response.headers

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        """Functional middleware blocks requests over limit."""
        # Create limiter with very low limits
        limiter = TokenBucketRateLimiter(rate=60, burst=1)

        request = MagicMock(spec=Request)
        request.headers.get.return_value = None
        request.client = MagicMock()
        request.client.host = "10.0.0.2"
        request.url.path = "/test"

        call_next = AsyncMock(return_value=MagicMock(headers={}))

        with patch("ww.api.middleware.rate_limit._rate_limiter", limiter):
            # First request succeeds
            response1 = await rate_limit_middleware(request, call_next)
            assert response1.status_code != 429

            # Second request fails
            response2 = await rate_limit_middleware(request, call_next)
            assert response2.status_code == 429


class TestRateLimitingUnderLoad:
    """Stress tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_handles_concurrent_requests(self):
        """Rate limiter handles concurrent requests correctly."""
        limiter = TokenBucketRateLimiter(rate=120, burst=50)

        async def make_request(client_num: int):
            client_id = f"client{client_num}"
            allowed, _ = await limiter.check_rate_limit(client_id)
            return (client_id, allowed)

        # 100 concurrent requests from 10 clients (10 each)
        tasks = [make_request(i % 10) for i in range(100)]
        results = await asyncio.gather(*tasks)

        # Each client gets 50 tokens, so all 10 requests per client should succeed
        allowed_by_client = {}
        for client_id, allowed in results:
            if client_id not in allowed_by_client:
                allowed_by_client[client_id] = 0
            if allowed:
                allowed_by_client[client_id] += 1

        # Each client should have succeeded for 10 requests (under 50 burst)
        for count in allowed_by_client.values():
            assert count == 10

    @pytest.mark.asyncio
    async def test_burst_handling(self):
        """Rate limiter handles burst correctly."""
        limiter = TokenBucketRateLimiter(rate=60, burst=10)

        # Burst of 10 requests should all succeed
        results = []
        for _ in range(10):
            allowed, _ = await limiter.check_rate_limit("client1")
            results.append(allowed)

        assert all(results)

        # 11th request should fail
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is False

    @pytest.mark.asyncio
    async def test_sustained_rate_vs_burst(self):
        """Burst allows temporary spikes but sustained rate enforces limit."""
        limiter = TokenBucketRateLimiter(rate=60, burst=100)  # 1 req/sec sustained

        # Initial burst of 100 should succeed
        for _ in range(100):
            allowed, _ = await limiter.check_rate_limit("client1")
            assert allowed is True

        # Next should fail (bucket empty)
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is False

        # Wait 1 second (1 token refills)
        await asyncio.sleep(1.1)

        # Should allow 1 more
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is True

        # But not 2
        allowed, _ = await limiter.check_rate_limit("client1")
        assert allowed is False


class TestRateLimitingIntegration:
    """Integration tests with FastAPI."""

    def test_integration_with_fastapi(self):
        """Rate limiting works end-to-end with FastAPI."""
        app = FastAPI()

        # Add rate limiting middleware with very low limit
        app.add_middleware(RateLimitMiddleware, rate=60, burst=2)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        # First two requests succeed
        response1 = client.get("/test")
        assert response1.status_code == 200
        assert "X-RateLimit-Limit" in response1.headers

        response2 = client.get("/test")
        assert response2.status_code == 200

        # Third request fails with 429
        response3 = client.get("/test")
        assert response3.status_code == 429
        assert "Retry-After" in response3.headers

    def test_rate_limit_headers_present(self):
        """Rate limit headers are present in responses."""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, rate=100, burst=200)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "100"

    def test_different_clients_isolated(self):
        """Different clients (IPs) have isolated rate limits."""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, rate=60, burst=1)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)

        # Client 1 exhausts limit
        response1 = client.get("/test")
        assert response1.status_code == 200

        # Client 1 blocked
        response2 = client.get("/test")
        assert response2.status_code == 429

        # Client 2 (different IP via header) should still work
        response3 = client.get("/test", headers={"X-Forwarded-For": "192.168.1.100"})
        assert response3.status_code == 200


class TestGetRateLimiter:
    """Tests for global rate limiter accessor."""

    def test_creates_limiter_on_first_call(self):
        """get_rate_limiter creates limiter on first call."""
        with patch("ww.api.middleware.rate_limit._rate_limiter", None):
            limiter = get_rate_limiter()
            assert limiter is not None
            assert isinstance(limiter, TokenBucketRateLimiter)

    def test_returns_same_instance(self):
        """get_rate_limiter returns same instance on subsequent calls."""
        with patch("ww.api.middleware.rate_limit._rate_limiter", None):
            limiter1 = get_rate_limiter()
            limiter2 = get_rate_limiter()
            assert limiter1 is limiter2
