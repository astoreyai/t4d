"""Fixtures for chaos testing."""
import pytest
import asyncio
import random


class ChaosMonkey:
    """Utility for injecting failures in tests."""

    def __init__(self):
        self.failure_rate = 0.0
        self.failure_type = Exception
        self.failure_message = "Chaos monkey strikes!"
        self.delay_ms = 0
        self.call_count = 0

    def set_failure_rate(self, rate: float, exception: type = Exception, message: str = "Chaos monkey strikes!"):
        """Set random failure rate (0.0 to 1.0)."""
        self.failure_rate = rate
        self.failure_type = exception
        self.failure_message = message

    def set_delay(self, ms: int):
        """Set artificial delay in milliseconds."""
        self.delay_ms = ms

    async def maybe_fail(self):
        """Maybe raise an exception based on failure rate."""
        self.call_count += 1
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)
        if random.random() < self.failure_rate:
            raise self.failure_type(self.failure_message)

    def maybe_fail_sync(self):
        """Synchronous version of maybe_fail."""
        self.call_count += 1
        if random.random() < self.failure_rate:
            raise self.failure_type(self.failure_message)

    def reset(self):
        """Reset chaos monkey state."""
        self.failure_rate = 0.0
        self.delay_ms = 0
        self.call_count = 0


@pytest.fixture
def chaos_monkey():
    """Provide a chaos monkey for tests."""
    monkey = ChaosMonkey()
    yield monkey
    monkey.reset()


@pytest.fixture
def deterministic_chaos():
    """Chaos monkey with deterministic failures (for reproducible tests)."""
    monkey = ChaosMonkey()
    # Seed random for reproducibility
    random.seed(42)
    yield monkey
    monkey.reset()
