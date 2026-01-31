"""Existing tests compatibility check (P7-03).

Note: Full 8,905 test pass requires P2-14 (saga removal) which is deferred.
This test verifies that all new T4DX, spiking, consolidation, and persistence
tests pass together as a regression gate.
"""

import subprocess
import sys

import pytest


class TestNewTestSuitePass:
    """Verify all new Phase 1-5 tests pass."""

    @pytest.mark.slow
    def test_storage_tests_pass(self):
        """All T4DX storage tests should pass."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/storage/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Storage tests failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.slow
    def test_consolidation_tests_pass(self):
        """All consolidation tests should pass."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/consolidation/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Consolidation tests failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.slow
    def test_persistence_tests_pass(self):
        """All persistence tests should pass."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/persistence/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Persistence tests failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.slow
    def test_qwen_tests_pass(self):
        """All Qwen integration tests should pass."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/qwen/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Qwen tests failed:\n{result.stdout}\n{result.stderr}"
