"""
Pytest configuration for benchmark tests.

Provides benchmark markers and shared fixtures for running benchmarks in CI.
"""

import sys
from pathlib import Path

import pytest

# Add benchmarks directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))


def pytest_configure(config):
    """Configure pytest markers for benchmarks."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark (run with -m benchmark)"
    )
    config.addinivalue_line(
        "markers", "bioplausibility: mark test as bioplausibility benchmark"
    )
    config.addinivalue_line(
        "markers", "memory: mark test as memory benchmark"
    )
    config.addinivalue_line(
        "markers", "retrieval: mark test as retrieval benchmark"
    )


@pytest.fixture
def benchmark_results():
    """Fixture to collect benchmark results."""
    return {
        "bioplausibility": [],
        "longmemeval": [],
        "dmr": [],
    }
