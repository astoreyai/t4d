"""
Rate limiting middleware for World Weaver API.

Implements token bucket algorithm for per-client rate limiting.

Phase 3B: Production-ready rate limiting with:
- Token bucket algorithm (handles bursts gracefully)
- Per-client isolation (IP or API key based)
- Standard 429 responses with retry headers
- X-RateLimit-* headers for client visibility
"""

import asyncio
import logging
import time
from typing import Dict, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket for rate limiting a single client.

    Algorithm:
    - Bucket holds tokens (max = burst capacity)
    - Tokens replenish at constant rate
    - Request consumes 1 token
    - Request rejected if insufficient tokens

    Args:
        rate: Tokens added per minute (requests/min)
        burst: Maximum bucket capacity (max burst size)
    """

    def __init__(self, rate: int, burst: int):
        self.rate = rate  # tokens per minute
        self.burst = burst  # max tokens
        self.tokens = float(burst)  # start full
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens available, False if rate limit exceeded
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time (rate is per minute)
            new_tokens = elapsed * (self.rate / 60.0)
            self.tokens = min(self.burst, self.tokens + new_tokens)
            self.last_update = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_state(self) -> tuple[float, float]:
        """Get current bucket state (tokens, capacity)."""
        return (self.tokens, self.burst)


class TokenBucketRateLimiter:
    """
    Rate limiting with token bucket algorithm.

    Features:
    - Per-client buckets (isolated limits)
    - Configurable rate and burst
    - Automatic bucket cleanup for inactive clients

    Args:
        rate: Requests per minute per client
        burst: Maximum burst size per client
    """

    def __init__(self, rate: int = 100, burst: int = 200):
        self.rate = rate
        self.burst = burst
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, client_id: str) -> tuple[bool, tuple[float, float]]:
        """
        Check if request is within rate limit.

        Args:
            client_id: Unique client identifier (IP or API key)

        Returns:
            Tuple of (allowed: bool, bucket_state: (tokens, capacity))
        """
        async with self._lock:
            if client_id not in self.buckets:
                self.buckets[client_id] = TokenBucket(self.rate, self.burst)

        bucket = self.buckets[client_id]
        allowed = await bucket.consume(1)
        state = bucket.get_state()

        return (allowed, state)

    def get_bucket_state(self, client_id: str) -> Optional[tuple[float, float]]:
        """Get current bucket state for a client (non-async)."""
        bucket = self.buckets.get(client_id)
        if bucket:
            return bucket.get_state()
        return None

    async def cleanup_inactive_buckets(self, max_age_seconds: float = 3600):
        """
        Remove buckets for inactive clients.

        Args:
            max_age_seconds: Remove buckets not accessed in this time
        """
        async with self._lock:
            now = time.monotonic()
            to_remove = []
            for client_id, bucket in self.buckets.items():
                if now - bucket.last_update > max_age_seconds:
                    to_remove.append(client_id)

            for client_id in to_remove:
                del self.buckets[client_id]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} inactive rate limit buckets")


# Global rate limiter instance
_rate_limiter: Optional[TokenBucketRateLimiter] = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        # Default: 100 req/min with 200 burst
        _rate_limiter = TokenBucketRateLimiter(rate=100, burst=200)
    return _rate_limiter


def get_client_id(request: Request) -> str:
    """
    Extract client identifier from request.

    Priority:
    1. X-API-Key header (if present)
    2. X-Forwarded-For header (if behind proxy)
    3. Client IP address

    Args:
        request: FastAPI request

    Returns:
        Unique client identifier
    """
    # Try API key first (most specific)
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key[:16]}"  # Use prefix to avoid logging full key

    # Try X-Forwarded-For (behind proxy)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take first IP (original client)
        return f"ip:{forwarded.split(',')[0].strip()}"

    # Fall back to direct client IP
    if request.client:
        return f"ip:{request.client.host}"

    # Fallback for missing client info
    return "unknown"


async def rate_limit_middleware(request: Request, call_next) -> Response:
    """
    FastAPI middleware function for rate limiting.

    Can be used with app.middleware("http") decorator.

    Args:
        request: Incoming request
        call_next: Next middleware/handler in chain

    Returns:
        Response (429 if rate limited, else handler response)
    """
    limiter = get_rate_limiter()
    client_id = get_client_id(request)

    allowed, (tokens, capacity) = await limiter.check_rate_limit(client_id)

    if not allowed:
        # Calculate retry-after (when bucket will have 1 token)
        tokens_needed = 1.0
        seconds_per_token = 60.0 / limiter.rate
        retry_after = int(tokens_needed * seconds_per_token) + 1

        logger.warning(
            f"Rate limit exceeded for client {client_id}",
            extra={
                "client_id": client_id,
                "path": request.url.path,
                "tokens_remaining": int(tokens),
            },
        )

        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please try again later.",
                "retry_after": retry_after,
            },
            headers={
                "X-RateLimit-Limit": str(limiter.rate),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                "Retry-After": str(retry_after),
            },
        )

    # Request allowed - process and add rate limit headers
    response = await call_next(request)

    # Add rate limit headers to successful response
    response.headers["X-RateLimit-Limit"] = str(limiter.rate)
    response.headers["X-RateLimit-Remaining"] = str(int(tokens))
    response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

    return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware as a class.

    Alternative to functional middleware for better typing and state management.

    Args:
        rate: Requests per minute per client (default: 100)
        burst: Maximum burst size per client (default: 200)
    """

    def __init__(self, app, rate: int = 100, burst: int = 200):
        super().__init__(app)
        self.limiter = TokenBucketRateLimiter(rate=rate, burst=burst)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        client_id = get_client_id(request)

        allowed, (tokens, capacity) = await self.limiter.check_rate_limit(client_id)

        if not allowed:
            # Calculate retry-after
            tokens_needed = 1.0
            seconds_per_token = 60.0 / self.limiter.rate
            retry_after = int(tokens_needed * seconds_per_token) + 1

            logger.warning(
                f"Rate limit exceeded for client {client_id}",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                    "tokens_remaining": int(tokens),
                },
            )

            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(self.limiter.rate),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                    "Retry-After": str(retry_after),
                },
            )

        # Request allowed
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.rate)
        response.headers["X-RateLimit-Remaining"] = str(int(tokens))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

        return response
