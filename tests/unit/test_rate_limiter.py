"""
Unit tests for RateLimiter security feature.

Tests the rate limiting functionality to prevent DoS attacks.
"""

import time
import threading
from collections import defaultdict


# Copy of RateLimiter class for standalone testing
class RateLimiter:
    """
    Rate limiter to prevent DoS attacks.

    Thread-safe rate limiting per session ID with sliding window.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, session_id: str) -> bool:
        """
        Check if request is allowed for session.

        Args:
            session_id: Session identifier

        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()
            # Remove expired timestamps
            self.requests[session_id] = [
                t for t in self.requests[session_id]
                if now - t < self.window
            ]
            # Check limit
            if len(self.requests[session_id]) >= self.max:
                return False
            # Add timestamp
            self.requests[session_id].append(now)
            return True

    def reset(self, session_id: str = None):
        """
        Reset rate limit for session(s).

        Args:
            session_id: Specific session to reset, or None for all
        """
        with self._lock:
            if session_id:
                self.requests.pop(session_id, None)
            else:
                self.requests.clear()

    def time_until_allowed(self, session_id: str) -> float:
        """
        Calculate seconds until next request allowed.

        Args:
            session_id: Session identifier

        Returns:
            Seconds until next request allowed
        """
        with self._lock:
            now = time.time()
            # Clean expired
            self.requests[session_id] = [
                t for t in self.requests[session_id]
                if now - t < self.window
            ]
            if len(self.requests[session_id]) < self.max:
                return 0.0
            # Oldest timestamp will expire first
            oldest = min(self.requests[session_id])
            return max(0.0, self.window - (now - oldest))


def test_rate_limiter_basic():
    """Test basic rate limiting functionality."""
    limiter = RateLimiter(max_requests=5, window_seconds=10)
    session_id = "test-session"

    # First 5 requests should be allowed
    for i in range(5):
        assert limiter.allow(session_id), f"Request {i+1} should be allowed"

    # 6th request should be blocked
    assert not limiter.allow(session_id), "Request 6 should be blocked"


def test_rate_limiter_multiple_sessions():
    """Test rate limiting with multiple sessions."""
    limiter = RateLimiter(max_requests=3, window_seconds=10)

    # Session 1
    for i in range(3):
        assert limiter.allow("session-1"), f"Session 1 request {i+1} should be allowed"

    # Session 2 should have independent limit
    for i in range(3):
        assert limiter.allow("session-2"), f"Session 2 request {i+1} should be allowed"

    # Both sessions should now be blocked
    assert not limiter.allow("session-1"), "Session 1 should be blocked"
    assert not limiter.allow("session-2"), "Session 2 should be blocked"


def test_rate_limiter_window_expiry():
    """Test that rate limit resets after window expires."""
    limiter = RateLimiter(max_requests=2, window_seconds=1)
    session_id = "test-session"

    # First 2 requests should be allowed
    assert limiter.allow(session_id)
    assert limiter.allow(session_id)

    # 3rd request should be blocked
    assert not limiter.allow(session_id)

    # Wait for window to expire
    time.sleep(1.1)

    # Should be allowed again
    assert limiter.allow(session_id), "Request should be allowed after window expiry"


def test_rate_limiter_reset():
    """Test manual reset functionality."""
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    session_id = "test-session"

    # Fill the limit
    assert limiter.allow(session_id)
    assert limiter.allow(session_id)
    assert not limiter.allow(session_id)

    # Reset specific session
    limiter.reset(session_id)

    # Should be allowed again
    assert limiter.allow(session_id), "Request should be allowed after reset"


def test_rate_limiter_reset_all():
    """Test reset all sessions."""
    limiter = RateLimiter(max_requests=1, window_seconds=10)

    # Fill limits for multiple sessions
    limiter.allow("session-1")
    limiter.allow("session-2")

    assert not limiter.allow("session-1")
    assert not limiter.allow("session-2")

    # Reset all
    limiter.reset()

    # Both should be allowed again
    assert limiter.allow("session-1"), "Session 1 should be allowed after reset all"
    assert limiter.allow("session-2"), "Session 2 should be allowed after reset all"


def test_rate_limiter_time_until_allowed():
    """Test calculation of retry-after time."""
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    session_id = "test-session"

    # Not rate limited yet
    assert limiter.time_until_allowed(session_id) == 0.0

    # Fill the limit
    limiter.allow(session_id)
    limiter.allow(session_id)

    # Should have retry time
    retry_after = limiter.time_until_allowed(session_id)
    assert retry_after > 0.0, "Should have positive retry time"
    assert retry_after <= 10.0, "Retry time should not exceed window"


def test_rate_limiter_sliding_window():
    """Test sliding window behavior."""
    limiter = RateLimiter(max_requests=3, window_seconds=2)
    session_id = "test-session"

    # Make 3 requests at t=0
    assert limiter.allow(session_id)
    assert limiter.allow(session_id)
    assert limiter.allow(session_id)

    # 4th should be blocked
    assert not limiter.allow(session_id)

    # Wait for 1 second (window still active)
    time.sleep(1.0)

    # Still blocked
    assert not limiter.allow(session_id)

    # Wait for window to fully expire
    time.sleep(1.1)

    # Should be allowed again
    assert limiter.allow(session_id), "Request should be allowed after window expiry"


def test_rate_limiter_stress():
    """Test rate limiter with many requests (stress test)."""
    limiter = RateLimiter(max_requests=100, window_seconds=60)
    session_id = "stress-test"

    # First 100 should pass
    for i in range(100):
        assert limiter.allow(session_id), f"Request {i+1} should be allowed"

    # 101st should be blocked
    assert not limiter.allow(session_id), "Request 101 should be blocked"


if __name__ == "__main__":
    print("Running rate limiter tests...")

    test_rate_limiter_basic()
    print("✓ Basic rate limiting")

    test_rate_limiter_multiple_sessions()
    print("✓ Multiple sessions")

    test_rate_limiter_window_expiry()
    print("✓ Window expiry")

    test_rate_limiter_reset()
    print("✓ Reset functionality")

    test_rate_limiter_reset_all()
    print("✓ Reset all sessions")

    test_rate_limiter_time_until_allowed()
    print("✓ Time until allowed calculation")

    test_rate_limiter_sliding_window()
    print("✓ Sliding window behavior")

    test_rate_limiter_stress()
    print("✓ Stress test (100 requests)")

    print("\nAll rate limiter tests passed!")
