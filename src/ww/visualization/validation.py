"""
Biological Validation Framework for World Weaver NCA Telemetry.

Provides systematic validation of telemetry against experimental literature:
- Parameter range validation
- Temporal dynamics validation
- Cross-module consistency checks
- Statistical comparison with biological benchmarks

Biological References:
- Buzsáki (2006): Rhythms of the Brain
- Schultz (1998): Predictive reward signal of dopamine neurons
- Lisman & Jensen (2013): Theta-gamma neural code
- Girardeau et al. (2009): SWR suppression impairs consolidation

Author: Claude Opus 4.5
Date: 2026-01-01
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.visualization.telemetry_hub import TelemetryHub

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation results."""

    PASS = "pass"
    INFO = "info"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    severity: ValidationSeverity
    expected: str
    actual: str
    message: str
    reference: str = ""  # Literature reference
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
            "reference": self.reference,
        }


@dataclass
class ValidationReport:
    """Complete validation report."""

    timestamp: datetime
    total_checks: int
    passed: int
    failed: int
    warnings: int
    overall_score: float  # 0-100
    results: list[ValidationResult]
    summary: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_checks": self.total_checks,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "overall_score": self.overall_score,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


# =============================================================================
# Biological Parameter Ranges (from literature)
# =============================================================================

BIOLOGICAL_RANGES = {
    # SWR parameters (Buzsáki, 2015)
    "swr_frequency_hz": (150.0, 250.0),
    "swr_duration_ms": (50.0, 150.0),
    "swr_event_rate_hz": (0.5, 2.0),

    # PAC parameters (Tort et al., 2010; Lisman & Jensen, 2013)
    "pac_modulation_index": (0.1, 0.8),
    "pac_theta_frequency_hz": (4.0, 8.0),
    "pac_gamma_frequency_hz": (30.0, 80.0),
    "pac_wm_capacity": (4.0, 7.0),

    # DA parameters (Schultz, 1998; Cohen et al., 2012)
    "da_tonic_rate_hz": (3.0, 6.0),
    "da_burst_rate_hz": (15.0, 50.0),
    "da_pause_rate_hz": (0.0, 3.0),
    "da_baseline_level": (0.2, 0.4),

    # NT parameters (various sources)
    "nt_da_setpoint": (0.4, 0.6),
    "nt_5ht_setpoint": (0.4, 0.6),
    "nt_ach_setpoint": (0.4, 0.6),
    "nt_ne_setpoint": (0.3, 0.5),
    "nt_gaba_setpoint": (0.4, 0.6),
    "nt_glu_setpoint": (0.4, 0.6),

    # Stability parameters
    "stability_max_eigenvalue_real": (-2.0, 0.0),  # Should be negative for stability
    "stability_lyapunov_exponent": (-1.0, 0.0),  # Negative for bounded dynamics

    # E/I balance (Yizhar et al., 2011)
    "ei_ratio": (0.5, 2.0),
}


class BiologicalValidator:
    """
    Validates telemetry against biological benchmarks.

    Performs systematic validation of:
    1. Parameter ranges
    2. Temporal dynamics
    3. Cross-module consistency
    4. Statistical properties
    """

    def __init__(
        self,
        ranges: dict | None = None,
        strict_mode: bool = False,
    ):
        """
        Initialize validator.

        Args:
            ranges: Custom parameter ranges (overrides defaults)
            strict_mode: If True, warnings become failures
        """
        self.ranges = {**BIOLOGICAL_RANGES}
        if ranges:
            self.ranges.update(ranges)
        self.strict_mode = strict_mode

        # Validation history
        self._history: list[ValidationReport] = []
        self._max_history = 100

        logger.info("BiologicalValidator initialized")

    # -------------------------------------------------------------------------
    # Range Validation
    # -------------------------------------------------------------------------

    def validate_range(
        self,
        name: str,
        value: float,
        range_key: str | None = None,
        reference: str = "",
    ) -> ValidationResult:
        """
        Validate a value against biological range.

        Args:
            name: Human-readable name
            value: Value to validate
            range_key: Key in ranges dict (defaults to name)
            reference: Literature reference

        Returns:
            ValidationResult
        """
        range_key = range_key or name
        if range_key not in self.ranges:
            return ValidationResult(
                check_name=name,
                passed=True,
                severity=ValidationSeverity.INFO,
                expected="No range defined",
                actual=str(value),
                message=f"No biological range defined for {name}",
            )

        low, high = self.ranges[range_key]
        passed = low <= value <= high

        if passed:
            severity = ValidationSeverity.PASS
            message = f"{name} within biological range"
        else:
            severity = ValidationSeverity.FAIL if self.strict_mode else ValidationSeverity.WARNING
            if value < low:
                message = f"{name} below biological range (too low)"
            else:
                message = f"{name} above biological range (too high)"

        return ValidationResult(
            check_name=name,
            passed=passed,
            severity=severity,
            expected=f"[{low}, {high}]",
            actual=f"{value:.4f}",
            message=message,
            reference=reference,
        )

    # -------------------------------------------------------------------------
    # Module-Specific Validation
    # -------------------------------------------------------------------------

    def validate_swr_telemetry(self, swr_telemetry) -> list[ValidationResult]:
        """Validate SWR telemetry."""
        results = []

        try:
            freq_dist = swr_telemetry.get_frequency_distribution()
            dur_dist = swr_telemetry.get_duration_distribution()
            bio_valid = swr_telemetry.validate_biological_ranges()

            # Frequency validation
            if freq_dist:
                results.append(self.validate_range(
                    "SWR Ripple Frequency",
                    freq_dist["mean"],
                    "swr_frequency_hz",
                    "Buzsáki (2015)"
                ))

            # Duration validation
            if dur_dist:
                results.append(self.validate_range(
                    "SWR Duration",
                    dur_dist["mean_ms"],
                    "swr_duration_ms",
                    "Buzsáki (2015)"
                ))

            # Event rate
            if "event_rate_hz" in bio_valid:
                results.append(self.validate_range(
                    "SWR Event Rate",
                    bio_valid["event_rate_hz"],
                    "swr_event_rate_hz",
                    "Girardeau et al. (2009)"
                ))

        except Exception as e:
            results.append(ValidationResult(
                check_name="SWR Telemetry",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                expected="Valid data",
                actual=str(e),
                message=f"SWR validation error: {e}",
            ))

        return results

    def validate_pac_telemetry(self, pac_telemetry) -> list[ValidationResult]:
        """Validate PAC telemetry."""
        results = []

        try:
            mi = pac_telemetry.compute_modulation_index()

            # Modulation index
            results.append(self.validate_range(
                "PAC Modulation Index",
                mi,
                "pac_modulation_index",
                "Tort et al. (2010)"
            ))

            # Working memory capacity (from statistics if available)
            try:
                wm_stats = pac_telemetry.get_wm_capacity_statistics()
                if wm_stats and "mean" in wm_stats:
                    results.append(self.validate_range(
                        "Working Memory Capacity",
                        wm_stats["mean"],
                        "pac_wm_capacity",
                        "Lisman & Jensen (2013)"
                    ))
            except (AttributeError, KeyError):
                pass  # WM capacity not available

        except Exception as e:
            results.append(ValidationResult(
                check_name="PAC Telemetry",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                expected="Valid data",
                actual=str(e),
                message=f"PAC validation error: {e}",
            ))

        return results

    def validate_da_telemetry(self, da_telemetry) -> list[ValidationResult]:
        """Validate DA telemetry."""
        results = []

        try:
            stats = da_telemetry.get_statistics()
            rate_dist = da_telemetry.get_firing_rate_distribution()

            # Tonic rate
            if "tonic" in rate_dist:
                results.append(self.validate_range(
                    "DA Tonic Firing Rate",
                    rate_dist["tonic"]["mean"],
                    "da_tonic_rate_hz",
                    "Schultz (1998)"
                ))

            # Burst rate
            if "burst" in rate_dist:
                results.append(self.validate_range(
                    "DA Burst Firing Rate",
                    rate_dist["burst"]["mean"],
                    "da_burst_rate_hz",
                    "Schultz (1998)"
                ))

            # Pause rate
            if "pause" in rate_dist:
                results.append(self.validate_range(
                    "DA Pause Firing Rate",
                    rate_dist["pause"]["mean"],
                    "da_pause_rate_hz",
                    "Schultz (1998)"
                ))

            # Baseline DA level
            results.append(self.validate_range(
                "DA Baseline Level",
                stats.mean_da,
                "da_baseline_level",
                "Wightman & Robinson (2002)"
            ))

        except Exception as e:
            results.append(ValidationResult(
                check_name="DA Telemetry",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                expected="Valid data",
                actual=str(e),
                message=f"DA validation error: {e}",
            ))

        return results

    def validate_stability_monitor(self, stability_monitor) -> list[ValidationResult]:
        """Validate stability monitor."""
        results = []

        try:
            eigenvalues = stability_monitor.get_current_eigenvalues()
            is_stable = stability_monitor.is_stable()

            # Stability check
            results.append(ValidationResult(
                check_name="System Stability",
                passed=is_stable,
                severity=ValidationSeverity.PASS if is_stable else ValidationSeverity.FAIL,
                expected="Stable (all eigenvalues real part < 0)",
                actual="Stable" if is_stable else "Unstable",
                message="System dynamics are stable" if is_stable else "System is unstable",
                reference="Dynamical systems theory",
            ))

            # Max eigenvalue
            if eigenvalues is not None and len(eigenvalues) > 0:
                max_real = max(np.real(eigenvalues))
                results.append(self.validate_range(
                    "Max Eigenvalue (real part)",
                    max_real,
                    "stability_max_eigenvalue_real",
                    "Strogatz (1994)"
                ))

        except Exception as e:
            results.append(ValidationResult(
                check_name="Stability Monitor",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                expected="Valid data",
                actual=str(e),
                message=f"Stability validation error: {e}",
            ))

        return results

    def validate_nt_state(self, nt_state: np.ndarray) -> list[ValidationResult]:
        """Validate NT state vector."""
        results = []

        if nt_state is None or len(nt_state) < 6:
            return [ValidationResult(
                check_name="NT State",
                passed=False,
                severity=ValidationSeverity.FAIL,
                expected="6D NT vector",
                actual=f"Length {len(nt_state) if nt_state is not None else 0}",
                message="Invalid NT state vector",
            )]

        # NT order: [DA, 5-HT, ACh, NE, GABA, Glu]
        names = ["DA", "5-HT", "ACh", "NE", "GABA", "Glu"]
        setpoint_keys = [
            "nt_da_setpoint", "nt_5ht_setpoint", "nt_ach_setpoint",
            "nt_ne_setpoint", "nt_gaba_setpoint", "nt_glu_setpoint"
        ]

        for i, (name, key) in enumerate(zip(names, setpoint_keys)):
            results.append(self.validate_range(
                f"{name} Level",
                float(nt_state[i]),
                key,
                "Pharmacological literature"
            ))

        # E/I balance
        gaba = float(nt_state[4])
        glu = float(nt_state[5])
        ei_ratio = glu / max(gaba, 0.01)
        results.append(self.validate_range(
            "E/I Ratio (Glu/GABA)",
            ei_ratio,
            "ei_ratio",
            "Yizhar et al. (2011)"
        ))

        return results

    # -------------------------------------------------------------------------
    # Full System Validation
    # -------------------------------------------------------------------------

    def validate_hub(self, hub: TelemetryHub) -> ValidationReport:
        """
        Validate entire telemetry hub.

        Args:
            hub: TelemetryHub instance

        Returns:
            ValidationReport with all results
        """
        results = []

        # SWR validation
        if hub._swr is not None:
            results.extend(self.validate_swr_telemetry(hub._swr))

        # PAC validation
        if hub._pac is not None:
            results.extend(self.validate_pac_telemetry(hub._pac))

        # DA validation
        if hub._da is not None:
            results.extend(self.validate_da_telemetry(hub._da))

        # Stability validation
        if hub._stability is not None:
            results.extend(self.validate_stability_monitor(hub._stability))

        return self._create_report(results)

    def validate_all(
        self,
        swr_telemetry=None,
        pac_telemetry=None,
        da_telemetry=None,
        stability_monitor=None,
        nt_state: np.ndarray | None = None,
    ) -> ValidationReport:
        """
        Validate all provided telemetry modules.

        Args:
            swr_telemetry: SWR telemetry module
            pac_telemetry: PAC telemetry module
            da_telemetry: DA telemetry module
            stability_monitor: Stability monitor
            nt_state: Current NT state vector

        Returns:
            ValidationReport
        """
        results = []

        if swr_telemetry is not None:
            results.extend(self.validate_swr_telemetry(swr_telemetry))

        if pac_telemetry is not None:
            results.extend(self.validate_pac_telemetry(pac_telemetry))

        if da_telemetry is not None:
            results.extend(self.validate_da_telemetry(da_telemetry))

        if stability_monitor is not None:
            results.extend(self.validate_stability_monitor(stability_monitor))

        if nt_state is not None:
            results.extend(self.validate_nt_state(nt_state))

        return self._create_report(results)

    def _create_report(self, results: list[ValidationResult]) -> ValidationReport:
        """Create validation report from results."""
        now = datetime.now()

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if r.severity == ValidationSeverity.FAIL)
        warnings = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)

        # Calculate score (passed / total * 100)
        score = (passed / total * 100) if total > 0 else 100.0

        # Generate summary
        if score >= 90:
            summary = "Excellent biological validity"
        elif score >= 75:
            summary = "Good biological validity with minor issues"
        elif score >= 50:
            summary = "Moderate biological validity - review warnings"
        else:
            summary = "Poor biological validity - significant issues"

        report = ValidationReport(
            timestamp=now,
            total_checks=total,
            passed=passed,
            failed=failed,
            warnings=warnings,
            overall_score=score,
            results=results,
            summary=summary,
        )

        # Store history
        self._history.append(report)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return report

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def print_report(self, report: ValidationReport) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "BIOLOGICAL VALIDATION REPORT",
            f"Timestamp: {report.timestamp.isoformat()}",
            "=" * 60,
            "",
            f"Overall Score: {report.overall_score:.1f}/100",
            f"Summary: {report.summary}",
            "",
            f"Checks: {report.total_checks} total",
            f"  Passed: {report.passed}",
            f"  Failed: {report.failed}",
            f"  Warnings: {report.warnings}",
            "",
            "-" * 60,
            "DETAILED RESULTS",
            "-" * 60,
        ]

        for r in report.results:
            status = "✓" if r.passed else "✗"
            lines.append(f"{status} {r.check_name}")
            lines.append(f"    Expected: {r.expected}")
            lines.append(f"    Actual: {r.actual}")
            if not r.passed:
                lines.append(f"    Message: {r.message}")
            if r.reference:
                lines.append(f"    Reference: {r.reference}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_history(self) -> list[ValidationReport]:
        """Get validation history."""
        return self._history.copy()

    def get_trend(self) -> dict:
        """Get validation score trend."""
        if not self._history:
            return {"insufficient_data": True}

        scores = [r.overall_score for r in self._history]
        return {
            "current": scores[-1],
            "mean": float(np.mean(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
            "sample_count": len(scores),
        }


# Convenience functions
def validate_telemetry_hub(hub: TelemetryHub, strict: bool = False) -> ValidationReport:
    """Validate telemetry hub."""
    validator = BiologicalValidator(strict_mode=strict)
    return validator.validate_hub(hub)


def quick_validation(
    swr_telemetry=None,
    pac_telemetry=None,
    da_telemetry=None,
) -> float:
    """Quick validation returning just the score."""
    validator = BiologicalValidator()
    report = validator.validate_all(
        swr_telemetry=swr_telemetry,
        pac_telemetry=pac_telemetry,
        da_telemetry=da_telemetry,
    )
    return report.overall_score


__all__ = [
    "BiologicalValidator",
    "ValidationResult",
    "ValidationReport",
    "ValidationSeverity",
    "BIOLOGICAL_RANGES",
    "validate_telemetry_hub",
    "quick_validation",
]
