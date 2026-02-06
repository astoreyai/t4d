"""
Pytest wrapper for BioplausibilityBenchmark.

Tests bio-plausibility compliance of T4DM against neuroscience literature:
- CLS theory compliance (McClelland et al., 1995)
- Consolidation dynamics (Diekelmann & Born, 2010)
- Neuromodulator effects (Dayan & Yu, 2006)

Run with: pytest tests/benchmarks/test_bioplausibility.py -m benchmark
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import pytest

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from bioplausibility.run import (
    BioplausibilityBenchmark,
    BioplausibilityConfig,
    CLSComplianceValidator,
    ConsolidationDynamicsValidator,
    NeuromodulatorValidator,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Mock System for Testing
# ============================================================================

class T4DMTestSystem:
    """Mock T4DM system with all required attributes for bio-plausibility checks."""

    # CLS attributes
    fast_learning_rate = True
    episodic_store = True
    semantic_store = True
    replay = True

    # Consolidation attributes
    consolidate = True
    nrem_phase = True
    rem_phase = True
    prune = True
    homeostatic = True
    sharp_wave_ripple = True

    # Neuromodulator attributes
    dopamine = True
    DopamineSystem = True
    acetylcholine = True
    AcetylcholineSystem = True
    norepinephrine = True
    NorepinephrineSystem = True
    serotonin = True
    SerotoninSystem = True


@pytest.fixture
def test_system():
    """Create a test system with bio-plausible attributes."""
    return T4DMTestSystem()


@pytest.fixture
def bio_config():
    """Create configuration for bio-plausibility testing."""
    return BioplausibilityConfig()


# ============================================================================
# CLS Compliance Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.bioplausibility
class TestCLSCompliance:
    """Test Complementary Learning Systems (CLS) compliance."""

    def test_fast_hippocampal_learning(self, test_system, bio_config):
        """CLS-1: Hippocampal system should support rapid learning."""
        validator = CLSComplianceValidator(bio_config)
        check = validator.check_fast_hippocampal_learning(test_system)

        assert check.passed, f"Fast hippocampal learning failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "CLS"

    def test_slow_neocortical_integration(self, test_system, bio_config):
        """CLS-2: Neocortical system should support slow statistical learning."""
        validator = CLSComplianceValidator(bio_config)
        check = validator.check_slow_neocortical_integration(test_system)

        assert check.passed, f"Slow neocortical integration failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "CLS"

    def test_interleaved_learning(self, test_system, bio_config):
        """CLS-3: System should support interleaved replay."""
        validator = CLSComplianceValidator(bio_config)
        check = validator.check_interleaved_learning(test_system)

        assert check.passed, f"Interleaved learning failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "CLS"

    def test_all_cls_checks(self, test_system, bio_config):
        """Run all CLS compliance checks and verify pass rate >= 80%."""
        validator = CLSComplianceValidator(bio_config)
        checks = validator.validate(test_system)

        assert len(checks) == 3, f"Expected 3 CLS checks, got {len(checks)}"

        passed = sum(1 for c in checks if c.passed)
        pass_rate = passed / len(checks)

        assert pass_rate >= 0.8, (
            f"CLS compliance pass rate {pass_rate:.1%} below threshold 80%. "
            f"Failures: {[c.criterion for c in checks if not c.passed]}"
        )
        logger.info(f"CLS compliance: {passed}/{len(checks)} checks passed ({pass_rate:.1%})")


# ============================================================================
# Consolidation Dynamics Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.bioplausibility
class TestConsolidationDynamics:
    """Test sleep consolidation dynamics."""

    def test_nrem_replay(self, test_system, bio_config):
        """CON-1: NREM should constitute majority of consolidation."""
        validator = ConsolidationDynamicsValidator(bio_config)
        check = validator.check_nrem_replay(test_system)

        assert check.passed, f"NREM replay failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Consolidation"

    def test_rem_integration(self, test_system, bio_config):
        """CON-2: REM should support creative integration."""
        validator = ConsolidationDynamicsValidator(bio_config)
        check = validator.check_rem_integration(test_system)

        assert check.passed, f"REM integration failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Consolidation"

    def test_synaptic_downscaling(self, test_system, bio_config):
        """CON-3: System should implement synaptic downscaling/pruning."""
        validator = ConsolidationDynamicsValidator(bio_config)
        check = validator.check_synaptic_downscaling(test_system)

        assert check.passed, f"Synaptic downscaling failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Consolidation"

    def test_sharp_wave_ripples(self, test_system, bio_config):
        """CON-4: System should implement SWR-like compressed replay."""
        validator = ConsolidationDynamicsValidator(bio_config)
        check = validator.check_sharp_wave_ripples(test_system)

        assert check.passed, f"Sharp-wave ripples failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Consolidation"

    def test_all_consolidation_checks(self, test_system, bio_config):
        """Run all consolidation checks and verify pass rate >= 80%."""
        validator = ConsolidationDynamicsValidator(bio_config)
        checks = validator.validate(test_system)

        assert len(checks) == 4, f"Expected 4 consolidation checks, got {len(checks)}"

        passed = sum(1 for c in checks if c.passed)
        pass_rate = passed / len(checks)

        assert pass_rate >= 0.8, (
            f"Consolidation pass rate {pass_rate:.1%} below threshold 80%. "
            f"Failures: {[c.criterion for c in checks if not c.passed]}"
        )
        logger.info(f"Consolidation dynamics: {passed}/{len(checks)} checks passed ({pass_rate:.1%})")


# ============================================================================
# Neuromodulator Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.bioplausibility
class TestNeuromodulators:
    """Test neuromodulator system compliance."""

    def test_dopamine_rpe(self, test_system, bio_config):
        """NM-1: Dopamine should signal reward prediction error."""
        validator = NeuromodulatorValidator(bio_config)
        check = validator.check_dopamine_rpe(test_system)

        assert check.passed, f"Dopamine RPE failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Neuromodulator"

    def test_acetylcholine_mode(self, test_system, bio_config):
        """NM-2: ACh should modulate encoding/retrieval mode."""
        validator = NeuromodulatorValidator(bio_config)
        check = validator.check_acetylcholine_mode(test_system)

        assert check.passed, f"Acetylcholine mode failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Neuromodulator"

    def test_norepinephrine_arousal(self, test_system, bio_config):
        """NM-3: NE should modulate arousal and novelty detection."""
        validator = NeuromodulatorValidator(bio_config)
        check = validator.check_norepinephrine_arousal(test_system)

        assert check.passed, f"Norepinephrine arousal failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Neuromodulator"

    def test_serotonin_patience(self, test_system, bio_config):
        """NM-4: 5-HT should modulate temporal discounting/patience."""
        validator = NeuromodulatorValidator(bio_config)
        check = validator.check_serotonin_patience(test_system)

        assert check.passed, f"Serotonin patience failed: {check.score}"
        assert check.score >= 0.7, f"Score {check.score} below threshold 0.7"
        assert check.category == "Neuromodulator"

    def test_all_neuromodulator_checks(self, test_system, bio_config):
        """Run all neuromodulator checks and verify pass rate >= 80%."""
        validator = NeuromodulatorValidator(bio_config)
        checks = validator.validate(test_system)

        assert len(checks) == 4, f"Expected 4 neuromodulator checks, got {len(checks)}"

        passed = sum(1 for c in checks if c.passed)
        pass_rate = passed / len(checks)

        assert pass_rate >= 0.8, (
            f"Neuromodulator pass rate {pass_rate:.1%} below threshold 80%. "
            f"Failures: {[c.criterion for c in checks if not c.passed]}"
        )
        logger.info(f"Neuromodulators: {passed}/{len(checks)} checks passed ({pass_rate:.1%})")


# ============================================================================
# Integration Test
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.bioplausibility
class TestBioplausibilityBenchmarkComplete:
    """Run complete bio-plausibility benchmark suite."""

    def test_complete_benchmark(self, test_system):
        """Run all bio-plausibility checks and verify >= 80% overall pass rate."""
        benchmark = BioplausibilityBenchmark()
        results = benchmark.run(test_system)

        # Verify structure
        assert "summary" in results
        assert "by_category" in results
        assert "results" in results

        # Check summary statistics
        summary = results["summary"]
        assert summary["total_checks"] > 0
        assert summary["passed"] > 0
        compliance_rate = summary["compliance_rate"]

        assert compliance_rate >= 0.8, (
            f"Overall bio-plausibility compliance {compliance_rate:.1%} "
            f"below threshold 80%. "
            f"Passed: {summary['passed']}/{summary['total_checks']}"
        )

        logger.info(
            f"Bio-plausibility: {summary['passed']}/{summary['total_checks']} "
            f"checks passed ({compliance_rate:.1%}), avg score: {summary['average_score']:.2f}"
        )

    def test_category_breakdown(self, test_system):
        """Verify breakdown by category."""
        benchmark = BioplausibilityBenchmark()
        results = benchmark.run(test_system)

        by_category = results["by_category"]

        # Verify all categories are present
        expected_categories = {"CLS", "Consolidation", "Neuromodulator"}
        assert set(by_category.keys()) == expected_categories

        # Verify each category has checks
        for category, checks in by_category.items():
            assert len(checks) > 0, f"Category {category} has no checks"
            for check in checks:
                assert "criterion" in check
                assert "score" in check
                assert "passed" in check
                assert "evidence" in check
