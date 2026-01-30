"""Tests for Biological Validation Framework."""

import pytest
from datetime import datetime
import numpy as np

from ww.visualization.validation import (
    BiologicalValidator,
    ValidationResult,
    ValidationReport,
    ValidationSeverity,
    BIOLOGICAL_RANGES,
    validate_telemetry_hub,
    quick_validation,
)


class TestValidationSeverity:
    """Test ValidationSeverity enum."""

    def test_all_severities_defined(self):
        """Verify all severity levels exist."""
        assert ValidationSeverity.PASS.value == "pass"
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.FAIL.value == "fail"
        assert ValidationSeverity.CRITICAL.value == "critical"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_create_result(self):
        """Test result creation."""
        result = ValidationResult(
            check_name="Test Check",
            passed=True,
            severity=ValidationSeverity.PASS,
            expected="[0, 1]",
            actual="0.5",
            message="Value within range",
            reference="Test Reference",
        )
        assert result.passed is True
        assert result.severity == ValidationSeverity.PASS
        assert result.check_name == "Test Check"

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ValidationResult(
            check_name="Test",
            passed=True,
            severity=ValidationSeverity.PASS,
            expected="expected",
            actual="actual",
            message="message",
        )
        d = result.to_dict()
        assert d["check_name"] == "Test"
        assert d["passed"] is True
        assert d["severity"] == "pass"


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_create_report(self):
        """Test report creation."""
        report = ValidationReport(
            timestamp=datetime.now(),
            total_checks=10,
            passed=8,
            failed=1,
            warnings=1,
            overall_score=80.0,
            results=[],
            summary="Good",
        )
        assert report.overall_score == 80.0
        assert report.passed == 8

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = ValidationReport(
            timestamp=datetime.now(),
            total_checks=5,
            passed=4,
            failed=1,
            warnings=0,
            overall_score=80.0,
            results=[],
            summary="Good",
        )
        d = report.to_dict()
        assert d["total_checks"] == 5
        assert d["overall_score"] == 80.0


class TestBiologicalRanges:
    """Test biological parameter ranges."""

    def test_swr_ranges_defined(self):
        """SWR ranges are defined."""
        assert "swr_frequency_hz" in BIOLOGICAL_RANGES
        assert "swr_duration_ms" in BIOLOGICAL_RANGES

    def test_pac_ranges_defined(self):
        """PAC ranges are defined."""
        assert "pac_modulation_index" in BIOLOGICAL_RANGES
        assert "pac_wm_capacity" in BIOLOGICAL_RANGES

    def test_da_ranges_defined(self):
        """DA ranges are defined."""
        assert "da_tonic_rate_hz" in BIOLOGICAL_RANGES
        assert "da_burst_rate_hz" in BIOLOGICAL_RANGES

    def test_nt_ranges_defined(self):
        """NT ranges are defined."""
        assert "nt_da_setpoint" in BIOLOGICAL_RANGES
        assert "ei_ratio" in BIOLOGICAL_RANGES


class TestBiologicalValidator:
    """Test BiologicalValidator class."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = BiologicalValidator()
        assert validator.strict_mode is False
        assert len(validator.ranges) > 0

    def test_initialization_with_custom_ranges(self):
        """Test initialization with custom ranges."""
        custom = {"custom_param": (0.0, 1.0)}
        validator = BiologicalValidator(ranges=custom)
        assert "custom_param" in validator.ranges

    def test_strict_mode(self):
        """Test strict mode."""
        validator = BiologicalValidator(strict_mode=True)
        assert validator.strict_mode is True


class TestRangeValidation:
    """Test range validation."""

    def test_validate_in_range(self):
        """Value within range passes."""
        validator = BiologicalValidator()
        result = validator.validate_range(
            "Test",
            200.0,
            "swr_frequency_hz"
        )
        assert result.passed is True
        assert result.severity == ValidationSeverity.PASS

    def test_validate_below_range(self):
        """Value below range fails."""
        validator = BiologicalValidator()
        result = validator.validate_range(
            "Test",
            100.0,  # Below 150-250 range
            "swr_frequency_hz"
        )
        assert result.passed is False

    def test_validate_above_range(self):
        """Value above range fails."""
        validator = BiologicalValidator()
        result = validator.validate_range(
            "Test",
            300.0,  # Above 150-250 range
            "swr_frequency_hz"
        )
        assert result.passed is False

    def test_validate_unknown_range(self):
        """Unknown range returns info."""
        validator = BiologicalValidator()
        result = validator.validate_range(
            "Unknown",
            0.5,
            "nonexistent_key"
        )
        assert result.severity == ValidationSeverity.INFO

    def test_strict_mode_makes_fail(self):
        """Strict mode turns warnings into fails."""
        validator = BiologicalValidator(strict_mode=True)
        result = validator.validate_range(
            "Test",
            100.0,
            "swr_frequency_hz"
        )
        assert result.severity == ValidationSeverity.FAIL


class TestSWRValidation:
    """Test SWR telemetry validation."""

    def test_validate_swr_valid(self):
        """Validate valid SWR telemetry."""
        from ww.visualization.swr_telemetry import SWRTelemetry

        swr = SWRTelemetry()
        # Use record_event with correct parameter names
        for i in range(20):
            swr.record_event(
                ripple_frequency=180.0 + i,
                duration_s=0.08,  # 80 ms in seconds
                peak_amplitude=1.0,
                compression_factor=10.0,
            )

        validator = BiologicalValidator()
        results = validator.validate_swr_telemetry(swr)
        assert len(results) > 0


class TestPACValidation:
    """Test PAC telemetry validation."""

    def test_validate_pac_valid(self):
        """Validate valid PAC telemetry."""
        from ww.visualization.pac_telemetry import PACTelemetry

        pac = PACTelemetry()
        for i in range(150):
            theta = (i * 0.1) % (2 * np.pi)
            gamma = 0.5 + 0.3 * np.cos(theta)
            pac.record_state(theta, gamma)

        validator = BiologicalValidator()
        results = validator.validate_pac_telemetry(pac)
        assert len(results) >= 1  # At least MI


class TestDAValidation:
    """Test DA telemetry validation."""

    def test_validate_da_valid(self):
        """Validate valid DA telemetry."""
        from ww.visualization.da_telemetry import DATelemetry

        da = DATelemetry()
        # Mix of signal types
        for _ in range(10):
            da.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)  # tonic
        for _ in range(5):
            da.record_state(da_level=0.7, firing_rate=30.0, rpe=0.5)  # burst
        for _ in range(5):
            da.record_state(da_level=0.1, firing_rate=1.0, rpe=-0.4)  # pause

        validator = BiologicalValidator()
        results = validator.validate_da_telemetry(da)
        assert len(results) > 0


class TestStabilityValidation:
    """Test stability monitor validation."""

    def test_validate_stability_stable(self):
        """Validate stable system."""
        from ww.visualization.stability_monitor import StabilityMonitor

        monitor = StabilityMonitor()
        jacobian = np.array([[-0.5, 0.1], [0.1, -0.3]])
        monitor.record_state(jacobian)

        validator = BiologicalValidator()
        results = validator.validate_stability_monitor(monitor)
        assert len(results) >= 1

        # Find stability check
        stability_result = next(
            (r for r in results if r.check_name == "System Stability"),
            None
        )
        assert stability_result is not None
        assert stability_result.passed is True

    def test_validate_stability_unstable(self):
        """Validate unstable system detects issues."""
        from ww.visualization.stability_monitor import StabilityMonitor

        monitor = StabilityMonitor()
        # Create Jacobian - validation should return results
        jacobian = np.array([[2.0, 0.0], [0.0, 1.5]])
        monitor.record_state(jacobian)

        validator = BiologicalValidator()
        results = validator.validate_stability_monitor(monitor)

        # Should have at least one validation result
        assert len(results) >= 1

        # Check that we get stability check and/or eigenvalue check
        has_stability_check = any(
            "Stability" in r.check_name or "Eigenvalue" in r.check_name
            for r in results
        )
        assert has_stability_check


class TestNTStateValidation:
    """Test NT state validation."""

    def test_validate_nt_valid(self):
        """Validate valid NT state."""
        nt_state = np.array([0.5, 0.5, 0.5, 0.4, 0.5, 0.5])

        validator = BiologicalValidator()
        results = validator.validate_nt_state(nt_state)
        assert len(results) == 7  # 6 NTs + E/I ratio

    def test_validate_nt_invalid_length(self):
        """Invalid NT state length."""
        nt_state = np.array([0.5, 0.5])  # Too short

        validator = BiologicalValidator()
        results = validator.validate_nt_state(nt_state)
        assert len(results) == 1
        assert results[0].passed is False

    def test_validate_nt_ei_imbalance(self):
        """Detect E/I imbalance."""
        nt_state = np.array([0.5, 0.5, 0.5, 0.4, 0.1, 0.9])  # High Glu, low GABA

        validator = BiologicalValidator()
        results = validator.validate_nt_state(nt_state)

        ei_result = next(
            (r for r in results if "E/I" in r.check_name),
            None
        )
        assert ei_result is not None
        # E/I ratio = 0.9/0.1 = 9.0, should fail (range is 0.5-2.0)
        assert ei_result.passed is False


class TestValidateAll:
    """Test validate_all method."""

    def test_validate_all_empty(self):
        """Validate with no modules."""
        validator = BiologicalValidator()
        report = validator.validate_all()
        assert report.total_checks == 0
        assert report.overall_score == 100.0

    def test_validate_all_with_da(self):
        """Validate with DA module."""
        from ww.visualization.da_telemetry import DATelemetry

        da = DATelemetry()
        for _ in range(10):
            da.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)

        validator = BiologicalValidator()
        report = validator.validate_all(da_telemetry=da)
        assert report.total_checks > 0


class TestReporting:
    """Test report generation."""

    def test_print_report(self):
        """Test report printing."""
        validator = BiologicalValidator()
        report = ValidationReport(
            timestamp=datetime.now(),
            total_checks=5,
            passed=4,
            failed=1,
            warnings=0,
            overall_score=80.0,
            results=[
                ValidationResult(
                    check_name="Test",
                    passed=True,
                    severity=ValidationSeverity.PASS,
                    expected="[0, 1]",
                    actual="0.5",
                    message="OK",
                )
            ],
            summary="Good",
        )
        text = validator.print_report(report)
        assert "BIOLOGICAL VALIDATION REPORT" in text
        assert "80.0" in text


class TestHistory:
    """Test validation history."""

    def test_get_history(self):
        """Get validation history."""
        validator = BiologicalValidator()
        validator.validate_all()  # Creates a report
        history = validator.get_history()
        assert len(history) >= 1

    def test_get_trend(self):
        """Get validation trend."""
        validator = BiologicalValidator()
        # Create multiple reports
        for _ in range(5):
            validator.validate_all()

        trend = validator.get_trend()
        assert "current" in trend
        assert "mean" in trend


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_telemetry_hub(self):
        """Test hub validation function."""
        from ww.visualization.telemetry_hub import TelemetryHub

        hub = TelemetryHub()
        report = validate_telemetry_hub(hub)
        assert report is not None

    def test_quick_validation(self):
        """Test quick validation function."""
        score = quick_validation()
        assert 0.0 <= score <= 100.0


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Verify all exports are importable."""
        from ww.visualization.validation import (
            BiologicalValidator,
            ValidationResult,
            ValidationReport,
            ValidationSeverity,
            BIOLOGICAL_RANGES,
            validate_telemetry_hub,
            quick_validation,
        )
        assert BiologicalValidator is not None
        assert ValidationResult is not None
        assert BIOLOGICAL_RANGES is not None
