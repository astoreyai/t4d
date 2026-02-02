"""
Multi-Scale Telemetry Hub for T4DM NCA.

Integrates all telemetry modules into a unified monitoring system:
- SWR (Sharp-Wave Ripple) telemetry
- PAC (Phase-Amplitude Coupling) telemetry
- DA (Dopamine) temporal structure
- Stability monitoring
- NT state dashboard
- Forward-Forward network telemetry (Phase 4)
- Capsule network telemetry (Phase 4)
- Glymphatic system telemetry (Phase 4)

Provides cross-scale correlation, unified dashboards, and validation.

Biological Rationale:
Neural systems operate across multiple timescales:
- Milliseconds: Spikes, synaptic transmission
- Seconds: Oscillations (theta, gamma, SWR)
- Minutes: Neuromodulator dynamics (DA, 5-HT)
- Hours: Consolidation, homeostatic plasticity
- Sleep cycles: Glymphatic clearance, memory pruning

This hub bridges these scales for integrated monitoring.

Author: Claude Opus 4.5
Date: 2026-01-01
Updated: 2026-01-04 (Phase 4 support)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from t4dm.visualization.capsule_visualizer import CapsuleVisualizer
    from t4dm.visualization.da_telemetry import DATelemetry

    # Phase 4 visualizers
    from t4dm.visualization.ff_visualizer import ForwardForwardVisualizer
    from t4dm.visualization.glymphatic_visualizer import GlymphaticVisualizer
    from t4dm.visualization.nt_state_dashboard import NTStateDashboard
    from t4dm.visualization.pac_telemetry import PACTelemetry
    from t4dm.visualization.stability_monitor import StabilityMonitor
    from t4dm.visualization.swr_telemetry import SWRTelemetry

logger = logging.getLogger(__name__)


class TimeScale(Enum):
    """Biological timescales for telemetry."""

    FAST = "fast"  # Milliseconds: spikes, SWR (150-250 Hz)
    OSCILLATORY = "oscillatory"  # Seconds: theta, gamma, PAC
    NEUROMODULATOR = "neuromodulator"  # Minutes: DA, 5-HT dynamics
    CONSOLIDATION = "consolidation"  # Hours: memory replay, homeostasis


@dataclass
class TelemetryConfig:
    """Configuration for telemetry hub."""

    # Enable/disable individual telemetry modules
    enable_swr: bool = True
    enable_pac: bool = True
    enable_da: bool = True
    enable_stability: bool = True
    enable_nt: bool = True

    # Phase 4 modules
    enable_ff: bool = True  # Forward-Forward network telemetry
    enable_capsule: bool = True  # Capsule network telemetry
    enable_glymphatic: bool = True  # Glymphatic system telemetry

    # Cross-scale correlation settings
    correlation_window: int = 100
    min_samples_for_correlation: int = 50

    # Alert thresholds
    critical_alert_threshold: int = 3  # Alerts before critical
    warning_persistence: int = 10  # Samples before clearing warning

    # Export settings
    export_format: str = "json"
    max_export_samples: int = 10000


@dataclass
class CrossScaleEvent:
    """Event detected across multiple timescales."""

    timestamp: datetime
    event_type: str
    scales_involved: list[TimeScale]
    metrics: dict[str, float]
    description: str
    severity: str = "info"  # info, warning, critical


@dataclass
class SystemHealth:
    """Overall system health summary."""

    timestamp: datetime
    overall_status: str  # healthy, warning, critical
    active_alerts: list[str]
    stability_score: float  # 0-1
    biological_validity: float  # 0-1
    modules_active: dict[str, bool]
    cross_scale_coherence: float  # 0-1


class TelemetryHub:
    """
    Multi-scale telemetry integration hub.

    Coordinates multiple telemetry modules:
    1. SWR Telemetry - Sharp-wave ripples (consolidation)
    2. PAC Telemetry - Theta-gamma coupling (working memory)
    3. DA Telemetry - Dopamine dynamics (reward/learning)
    4. Stability Monitor - System stability (bifurcations)
    5. NT State Dashboard - Neuromodulator balance

    Provides unified monitoring, cross-scale correlation, and alerts.
    """

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        swr_telemetry: SWRTelemetry | None = None,
        pac_telemetry: PACTelemetry | None = None,
        da_telemetry: DATelemetry | None = None,
        stability_monitor: StabilityMonitor | None = None,
        nt_dashboard: NTStateDashboard | None = None,
        ff_visualizer: ForwardForwardVisualizer | None = None,
        capsule_visualizer: CapsuleVisualizer | None = None,
        glymphatic_visualizer: GlymphaticVisualizer | None = None,
    ):
        """
        Initialize telemetry hub.

        Args:
            config: Hub configuration
            swr_telemetry: SWR telemetry module
            pac_telemetry: PAC telemetry module
            da_telemetry: DA telemetry module
            stability_monitor: Stability monitor module
            nt_dashboard: NT state dashboard module
            ff_visualizer: Forward-Forward network visualizer (Phase 4)
            capsule_visualizer: Capsule network visualizer (Phase 4)
            glymphatic_visualizer: Glymphatic system visualizer (Phase 4)
        """
        self.config = config or TelemetryConfig()

        # Core telemetry modules
        self._swr = swr_telemetry
        self._pac = pac_telemetry
        self._da = da_telemetry
        self._stability = stability_monitor
        self._nt = nt_dashboard

        # Phase 4 modules
        self._ff = ff_visualizer
        self._capsule = capsule_visualizer
        self._glymphatic = glymphatic_visualizer

        # Cross-scale events
        self._events: list[CrossScaleEvent] = []
        self._max_events = 1000

        # Active alerts
        self._active_alerts: dict[str, int] = {}  # alert -> persistence counter

        # Correlation history
        self._correlation_history: list[dict] = []

        # Health snapshots
        self._health_history: list[SystemHealth] = []
        self._max_health_history = 100

        logger.info("TelemetryHub initialized (Phase 4 enabled)")

    # -------------------------------------------------------------------------
    # Module Registration
    # -------------------------------------------------------------------------

    def register_swr(self, swr_telemetry: SWRTelemetry) -> None:
        """Register SWR telemetry module."""
        self._swr = swr_telemetry
        logger.info("SWR telemetry registered")

    def register_pac(self, pac_telemetry: PACTelemetry) -> None:
        """Register PAC telemetry module."""
        self._pac = pac_telemetry
        logger.info("PAC telemetry registered")

    def register_da(self, da_telemetry: DATelemetry) -> None:
        """Register DA telemetry module."""
        self._da = da_telemetry
        logger.info("DA telemetry registered")

    def register_stability(self, stability_monitor: StabilityMonitor) -> None:
        """Register stability monitor."""
        self._stability = stability_monitor
        logger.info("Stability monitor registered")

    def register_nt(self, nt_dashboard: NTStateDashboard) -> None:
        """Register NT state dashboard."""
        self._nt = nt_dashboard
        logger.info("NT dashboard registered")

    # Phase 4 Registration

    def register_ff(self, ff_visualizer: ForwardForwardVisualizer) -> None:
        """
        Register Forward-Forward network visualizer.

        The FF visualizer tracks:
        - Layer goodness G(h) = sum(h^2)
        - Positive/negative phase separation
        - Threshold learning dynamics
        - FF-NCA coupling metrics

        Args:
            ff_visualizer: ForwardForwardVisualizer instance
        """
        self._ff = ff_visualizer
        logger.info("Forward-Forward visualizer registered")

    def register_capsule(self, capsule_visualizer: CapsuleVisualizer) -> None:
        """
        Register Capsule network visualizer.

        The Capsule visualizer tracks:
        - Pose vectors and activation probabilities
        - Dynamic routing iterations
        - Part-whole hierarchies
        - Capsule-NCA coupling metrics

        Args:
            capsule_visualizer: CapsuleVisualizer instance
        """
        self._capsule = capsule_visualizer
        logger.info("Capsule visualizer registered")

    def register_glymphatic(
        self, glymphatic_visualizer: GlymphaticVisualizer
    ) -> None:
        """
        Register Glymphatic system visualizer.

        The Glymphatic visualizer tracks:
        - Sleep stage transitions
        - AQP4 channel polarization
        - Waste clearance rates
        - Memory pruning decisions
        - SWR-replay coupling

        Args:
            glymphatic_visualizer: GlymphaticVisualizer instance
        """
        self._glymphatic = glymphatic_visualizer
        logger.info("Glymphatic visualizer registered")

    # -------------------------------------------------------------------------
    # Unified Recording
    # -------------------------------------------------------------------------

    def record_state(
        self,
        nt_state: np.ndarray | None = None,
        swr_state: Any | None = None,
        oscillator_state: Any | None = None,
        vta_state: Any | None = None,
        jacobian: np.ndarray | None = None,
        ff_state: Any | None = None,
        capsule_state: Any | None = None,
        glymphatic_state: Any | None = None,
    ) -> dict:
        """
        Record state across all registered modules.

        Args:
            nt_state: 6D NT concentration vector
            swr_state: SWR coupling state
            oscillator_state: Oscillator state (theta/gamma)
            vta_state: VTA circuit state
            jacobian: System Jacobian for stability
            ff_state: Forward-Forward layer state (Phase 4)
            capsule_state: Capsule network state (Phase 4)
            glymphatic_state: Glymphatic system state (Phase 4)

        Returns:
            Dict of recorded snapshots by module
        """
        results = {}
        now = datetime.now()

        # Record to SWR telemetry
        if self._swr is not None and swr_state is not None:
            try:
                event = self._swr.record_swr_event(swr_state)
                results["swr"] = event
            except Exception as e:
                logger.warning(f"SWR recording failed: {e}")

        # Record to PAC telemetry
        if self._pac is not None and oscillator_state is not None:
            try:
                # Extract theta phase and gamma amplitude from oscillator
                theta_phase = getattr(oscillator_state, "theta_phase", 0.0)
                gamma_amp = getattr(oscillator_state, "gamma_amplitude", 0.5)
                snapshot = self._pac.record_state(theta_phase, gamma_amp)
                results["pac"] = snapshot
            except Exception as e:
                logger.warning(f"PAC recording failed: {e}")

        # Record to DA telemetry
        if self._da is not None and vta_state is not None:
            try:
                snapshot = self._da.record_from_vta(vta_state)
                results["da"] = snapshot
            except Exception as e:
                logger.warning(f"DA recording failed: {e}")

        # Record to stability monitor
        if self._stability is not None and jacobian is not None:
            try:
                snapshot = self._stability.record_state(jacobian)
                results["stability"] = snapshot
            except Exception as e:
                logger.warning(f"Stability recording failed: {e}")

        # Record to NT dashboard
        if self._nt is not None and nt_state is not None:
            try:
                snapshot = self._nt.record_state(nt_state)
                results["nt"] = snapshot
            except Exception as e:
                logger.warning(f"NT recording failed: {e}")

        # ---------------------------------------------------------------------
        # Phase 4 Recording
        # ---------------------------------------------------------------------

        # Record Forward-Forward state
        if self._ff is not None and ff_state is not None:
            try:
                snapshot = self._ff.record_state(ff_state)
                results["ff"] = snapshot
            except Exception as e:
                logger.warning(f"Forward-Forward recording failed: {e}")

        # Record Capsule network state
        if self._capsule is not None and capsule_state is not None:
            try:
                snapshot = self._capsule.record_state(capsule_state)
                results["capsule"] = snapshot
            except Exception as e:
                logger.warning(f"Capsule recording failed: {e}")

        # Record Glymphatic system state
        if self._glymphatic is not None and glymphatic_state is not None:
            try:
                snapshot = self._glymphatic.record_state(glymphatic_state)
                results["glymphatic"] = snapshot
            except Exception as e:
                logger.warning(f"Glymphatic recording failed: {e}")

        # Check for cross-scale events
        self._detect_cross_scale_events(results)

        return results

    # -------------------------------------------------------------------------
    # Cross-Scale Analysis
    # -------------------------------------------------------------------------

    def _detect_cross_scale_events(self, snapshots: dict) -> None:
        """Detect events spanning multiple timescales."""
        now = datetime.now()
        events_detected = []

        # Check SWR + DA correlation (consolidation + reward)
        if "swr" in snapshots and "da" in snapshots:
            swr = snapshots["swr"]
            da = snapshots["da"]
            if swr is not None and da is not None:
                # Check if SWR during high DA (reward-driven consolidation)
                if hasattr(da, "da_level") and da.da_level > 0.6:
                    events_detected.append(
                        CrossScaleEvent(
                            timestamp=now,
                            event_type="reward_consolidation",
                            scales_involved=[
                                TimeScale.FAST,
                                TimeScale.NEUROMODULATOR,
                            ],
                            metrics={
                                "da_level": da.da_level,
                                "swr_detected": True,
                            },
                            description="SWR during elevated DA - reward memory consolidation",
                            severity="info",
                        )
                    )

        # Check PAC + Stability (oscillation coherence)
        if "pac" in snapshots and "stability" in snapshots:
            pac = snapshots["pac"]
            stability = snapshots["stability"]
            if pac is not None and stability is not None:
                mi = getattr(pac, "modulation_index", 0.0)
                is_stable = getattr(stability, "is_stable", True)
                if mi > 0.5 and not is_stable:
                    events_detected.append(
                        CrossScaleEvent(
                            timestamp=now,
                            event_type="oscillation_instability",
                            scales_involved=[
                                TimeScale.OSCILLATORY,
                                TimeScale.NEUROMODULATOR,
                            ],
                            metrics={
                                "modulation_index": mi,
                                "is_stable": is_stable,
                            },
                            description="Strong PAC during unstable dynamics",
                            severity="warning",
                        )
                    )

        # ---------------------------------------------------------------------
        # Phase 4: Cross-Scale Events
        # ---------------------------------------------------------------------

        # FF + Glymphatic coupling (learning + sleep clearance)
        if "ff" in snapshots and "glymphatic" in snapshots:
            ff = snapshots["ff"]
            gly = snapshots["glymphatic"]
            if ff is not None and gly is not None:
                goodness = getattr(ff, "goodness", 0.0)
                sleep_stage = getattr(gly, "sleep_stage", "wake")
                clearance_rate = getattr(gly, "clearance_rate", 0.0)

                # Detect sleep-gated FF negative generation
                if sleep_stage in ["NREM3", "REM"] and goodness < 0:
                    events_detected.append(
                        CrossScaleEvent(
                            timestamp=now,
                            event_type="sleep_negative_generation",
                            scales_involved=[
                                TimeScale.CONSOLIDATION,
                                TimeScale.OSCILLATORY,
                            ],
                            metrics={
                                "goodness": goodness,
                                "sleep_stage": sleep_stage,
                                "clearance_rate": clearance_rate,
                            },
                            description="FF negative phase during deep sleep - memory pruning",
                            severity="info",
                        )
                    )

        # Capsule + NCA coupling (pose vectors + NT state)
        if "capsule" in snapshots and "nt" in snapshots:
            capsule = snapshots["capsule"]
            nt = snapshots["nt"]
            if capsule is not None and nt is not None:
                routing_convergence = getattr(capsule, "routing_convergence", 1.0)
                ach_level = getattr(nt, "ach", 0.5)

                # Detect ACh-modulated routing
                if routing_convergence < 0.3 and ach_level > 0.7:
                    events_detected.append(
                        CrossScaleEvent(
                            timestamp=now,
                            event_type="ach_routing_enhancement",
                            scales_involved=[
                                TimeScale.FAST,
                                TimeScale.NEUROMODULATOR,
                            ],
                            metrics={
                                "routing_convergence": routing_convergence,
                                "ach_level": ach_level,
                            },
                            description="High ACh enhancing capsule routing - encoding mode",
                            severity="info",
                        )
                    )

        # Glymphatic + SWR coupling (clearance + replay)
        if "glymphatic" in snapshots and "swr" in snapshots:
            gly = snapshots["glymphatic"]
            swr = snapshots["swr"]
            if gly is not None and swr is not None:
                clearance = getattr(gly, "clearance_rate", 0.0)
                replay_count = getattr(swr, "replay_count", 0)

                # Detect coupled consolidation + clearance
                if clearance > 0.5 and replay_count > 5:
                    events_detected.append(
                        CrossScaleEvent(
                            timestamp=now,
                            event_type="replay_clearance_coupling",
                            scales_involved=[
                                TimeScale.FAST,
                                TimeScale.CONSOLIDATION,
                            ],
                            metrics={
                                "clearance_rate": clearance,
                                "replay_count": replay_count,
                            },
                            description="Active SWR replay with high glymphatic clearance",
                            severity="info",
                        )
                    )

        # Store events
        for event in events_detected:
            self._events.append(event)
            if event.severity in ["warning", "critical"]:
                self._active_alerts[event.event_type] = self.config.warning_persistence

        # Trim events list
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def compute_cross_scale_correlation(self) -> dict:
        """
        Compute correlations between telemetry streams.

        Returns:
            Dict of correlation coefficients between streams
        """
        correlations = {}

        # Get traces from each module
        da_trace = self._get_da_trace()
        pac_trace = self._get_pac_trace()
        stability_trace = self._get_stability_trace()

        min_len = min(len(da_trace), len(pac_trace), len(stability_trace))

        if min_len < self.config.min_samples_for_correlation:
            return {"insufficient_data": True}

        # Truncate to same length
        da_trace = da_trace[-min_len:]
        pac_trace = pac_trace[-min_len:]
        stability_trace = stability_trace[-min_len:]

        # Compute correlations
        if len(da_trace) > 1:
            # DA - PAC correlation
            correlations["da_pac"] = float(
                np.corrcoef(da_trace, pac_trace)[0, 1]
            ) if np.std(da_trace) > 0 and np.std(pac_trace) > 0 else 0.0

            # DA - Stability correlation
            correlations["da_stability"] = float(
                np.corrcoef(da_trace, stability_trace)[0, 1]
            ) if np.std(da_trace) > 0 and np.std(stability_trace) > 0 else 0.0

            # PAC - Stability correlation
            correlations["pac_stability"] = float(
                np.corrcoef(pac_trace, stability_trace)[0, 1]
            ) if np.std(pac_trace) > 0 and np.std(stability_trace) > 0 else 0.0

        correlations["sample_count"] = min_len

        # Store history
        self._correlation_history.append({
            "timestamp": datetime.now().isoformat(),
            **correlations,
        })
        if len(self._correlation_history) > 100:
            self._correlation_history = self._correlation_history[-100:]

        return correlations

    def _get_da_trace(self) -> list[float]:
        """Get DA level trace."""
        if self._da is None:
            return []
        return [s.da_level for s in self._da._snapshots]

    def _get_pac_trace(self) -> list[float]:
        """Get PAC modulation index trace."""
        if self._pac is None:
            return []
        return [s.modulation_index for s in self._pac._snapshots]

    def _get_stability_trace(self) -> list[float]:
        """Get stability margin trace."""
        if self._stability is None:
            return []
        return self._stability.get_stability_margin_trace()

    # -------------------------------------------------------------------------
    # Health Monitoring
    # -------------------------------------------------------------------------

    def get_system_health(self) -> SystemHealth:
        """
        Compute overall system health.

        Returns:
            SystemHealth summary
        """
        now = datetime.now()

        # Collect active alerts
        active = []
        to_remove = []
        for alert, count in self._active_alerts.items():
            if count > 0:
                active.append(alert)
                self._active_alerts[alert] = count - 1
            else:
                to_remove.append(alert)
        for alert in to_remove:
            del self._active_alerts[alert]

        # Determine overall status
        if len(active) >= self.config.critical_alert_threshold:
            status = "critical"
        elif len(active) > 0:
            status = "warning"
        else:
            status = "healthy"

        # Compute stability score
        stability_score = self._compute_stability_score()

        # Compute biological validity
        bio_validity = self._compute_biological_validity()

        # Module status
        modules = {
            "swr": self._swr is not None and self.config.enable_swr,
            "pac": self._pac is not None and self.config.enable_pac,
            "da": self._da is not None and self.config.enable_da,
            "stability": self._stability is not None and self.config.enable_stability,
            "nt": self._nt is not None and self.config.enable_nt,
            # Phase 4
            "ff": self._ff is not None and self.config.enable_ff,
            "capsule": self._capsule is not None and self.config.enable_capsule,
            "glymphatic": self._glymphatic is not None and self.config.enable_glymphatic,
        }

        # Cross-scale coherence
        correlations = self.compute_cross_scale_correlation()
        coherence = self._compute_coherence(correlations)

        health = SystemHealth(
            timestamp=now,
            overall_status=status,
            active_alerts=active,
            stability_score=stability_score,
            biological_validity=bio_validity,
            modules_active=modules,
            cross_scale_coherence=coherence,
        )

        # Store history
        self._health_history.append(health)
        if len(self._health_history) > self._max_health_history:
            self._health_history = self._health_history[-self._max_health_history:]

        return health

    def _compute_stability_score(self) -> float:
        """Compute stability score from stability monitor."""
        if self._stability is None:
            return 1.0  # Assume stable if not monitored

        try:
            if self._stability.is_stable():
                return 1.0
            # Get current eigenvalue spectrum
            eigenvalues = self._stability.get_current_eigenvalues()
            if eigenvalues is None:
                return 0.5
            max_real = max(np.real(eigenvalues))
            # Score based on how far from instability
            return float(np.clip(1.0 - max_real, 0.0, 1.0))
        except Exception:
            return 0.5

    def _compute_biological_validity(self) -> float:
        """Compute biological validity score across modules."""
        validity_scores = []

        # DA validation
        if self._da is not None:
            try:
                validation = self._da.validate_biological_ranges()
                valid_count = sum(
                    1 for k, v in validation.items()
                    if k.endswith("_in_range") and v is True
                )
                total = sum(1 for k in validation.keys() if k.endswith("_in_range"))
                if total > 0:
                    validity_scores.append(valid_count / total)
            except Exception:
                pass

        # PAC validation (MI should be 0.3-0.7)
        if self._pac is not None:
            try:
                mi = self._pac.compute_modulation_index()
                if 0.3 <= mi <= 0.7:
                    validity_scores.append(1.0)
                elif 0.1 <= mi <= 0.9:
                    validity_scores.append(0.5)
                else:
                    validity_scores.append(0.0)
            except Exception:
                pass

        # SWR validation
        if self._swr is not None:
            try:
                validation = self._swr.validate_biological_ranges()
                valid_count = sum(
                    1 for k, v in validation.items()
                    if k.endswith("_in_range") and v is True
                )
                total = sum(1 for k in validation.keys() if k.endswith("_in_range"))
                if total > 0:
                    validity_scores.append(valid_count / total)
            except Exception:
                pass

        if not validity_scores:
            return 1.0  # Assume valid if nothing to validate

        return float(np.mean(validity_scores))

    def _compute_coherence(self, correlations: dict) -> float:
        """Compute cross-scale coherence from correlations."""
        if correlations.get("insufficient_data", False):
            return 0.5  # Unknown

        corr_values = [
            abs(correlations.get("da_pac", 0.0)),
            abs(correlations.get("da_stability", 0.0)),
            abs(correlations.get("pac_stability", 0.0)),
        ]

        # High correlations indicate coherent dynamics
        return float(np.mean(corr_values))

    # -------------------------------------------------------------------------
    # Statistics & Export
    # -------------------------------------------------------------------------

    def get_summary_statistics(self) -> dict:
        """Get summary statistics from all modules."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "modules": {},
        }

        if self._swr is not None:
            try:
                stats["modules"]["swr"] = self._swr.get_replay_statistics()
            except Exception as e:
                stats["modules"]["swr"] = {"error": str(e)}

        if self._pac is not None:
            try:
                stats["modules"]["pac"] = {
                    "modulation_index": self._pac.compute_modulation_index(),
                    "preferred_phase": self._pac.find_preferred_phase(),
                    "wm_capacity": self._pac.estimate_wm_capacity(),
                }
            except Exception as e:
                stats["modules"]["pac"] = {"error": str(e)}

        if self._da is not None:
            try:
                da_stats = self._da.get_statistics()
                stats["modules"]["da"] = {
                    "mean_da": da_stats.mean_da,
                    "phasic_bursts": da_stats.phasic_burst_count,
                    "phasic_pauses": da_stats.phasic_pause_count,
                    "mean_rpe": da_stats.mean_rpe,
                }
            except Exception as e:
                stats["modules"]["da"] = {"error": str(e)}

        if self._stability is not None:
            try:
                stab_stats = self._stability.compute_stability_statistics()
                stats["modules"]["stability"] = stab_stats
            except Exception as e:
                stats["modules"]["stability"] = {"error": str(e)}

        # Phase 4 modules
        if self._ff is not None:
            try:
                stats["modules"]["ff"] = {
                    "mean_goodness": self._ff.get_mean_goodness(),
                    "pos_neg_separation": self._ff.get_separation_score(),
                    "threshold_range": self._ff.get_threshold_range(),
                    "layer_count": self._ff.get_layer_count(),
                }
            except Exception as e:
                stats["modules"]["ff"] = {"error": str(e)}

        if self._capsule is not None:
            try:
                stats["modules"]["capsule"] = {
                    "routing_convergence": self._capsule.get_routing_convergence(),
                    "mean_activation": self._capsule.get_mean_activation(),
                    "pose_variance": self._capsule.get_pose_variance(),
                    "hierarchy_depth": self._capsule.get_hierarchy_depth(),
                }
            except Exception as e:
                stats["modules"]["capsule"] = {"error": str(e)}

        if self._glymphatic is not None:
            try:
                stats["modules"]["glymphatic"] = {
                    "clearance_rate": self._glymphatic.get_clearance_rate(),
                    "aqp4_polarization": self._glymphatic.get_aqp4_polarization(),
                    "sleep_stage": self._glymphatic.get_current_sleep_stage(),
                    "pruning_count": self._glymphatic.get_pruning_count(),
                }
            except Exception as e:
                stats["modules"]["glymphatic"] = {"error": str(e)}

        # Add cross-scale info
        stats["cross_scale"] = {
            "correlations": self.compute_cross_scale_correlation(),
            "recent_events": len(self._events),
            "phase4_events": sum(
                1 for e in self._events[-100:]
                if e.event_type in [
                    "sleep_negative_generation",
                    "ach_routing_enhancement",
                    "replay_clearance_coupling",
                ]
            ),
        }

        # Add health
        health = self.get_system_health()
        stats["health"] = {
            "status": health.overall_status,
            "stability_score": health.stability_score,
            "biological_validity": health.biological_validity,
            "coherence": health.cross_scale_coherence,
        }

        return stats

    def export_all_data(self) -> dict:
        """Export data from all modules."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "enable_swr": self.config.enable_swr,
                "enable_pac": self.config.enable_pac,
                "enable_da": self.config.enable_da,
                "enable_stability": self.config.enable_stability,
                "enable_nt": self.config.enable_nt,
                # Phase 4
                "enable_ff": self.config.enable_ff,
                "enable_capsule": self.config.enable_capsule,
                "enable_glymphatic": self.config.enable_glymphatic,
            },
            "modules": {},
            "cross_scale_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "scales": [s.value for s in e.scales_involved],
                    "metrics": e.metrics,
                    "description": e.description,
                    "severity": e.severity,
                }
                for e in self._events[-100:]  # Last 100 events
            ],
            "correlation_history": self._correlation_history[-50:],
            "health_history": [
                {
                    "timestamp": h.timestamp.isoformat(),
                    "status": h.overall_status,
                    "stability": h.stability_score,
                    "validity": h.biological_validity,
                }
                for h in self._health_history[-50:]
            ],
        }

        # Export each module's data
        if self._swr is not None:
            try:
                data["modules"]["swr"] = self._swr.export_data()
            except Exception as e:
                data["modules"]["swr"] = {"error": str(e)}

        if self._pac is not None:
            try:
                data["modules"]["pac"] = self._pac.export_data()
            except Exception as e:
                data["modules"]["pac"] = {"error": str(e)}

        if self._da is not None:
            try:
                data["modules"]["da"] = self._da.export_data()
            except Exception as e:
                data["modules"]["da"] = {"error": str(e)}

        if self._stability is not None:
            try:
                data["modules"]["stability"] = self._stability.export_data()
            except Exception as e:
                data["modules"]["stability"] = {"error": str(e)}

        # Phase 4 exports
        if self._ff is not None:
            try:
                data["modules"]["ff"] = self._ff.export_data()
            except Exception as e:
                data["modules"]["ff"] = {"error": str(e)}

        if self._capsule is not None:
            try:
                data["modules"]["capsule"] = self._capsule.export_data()
            except Exception as e:
                data["modules"]["capsule"] = {"error": str(e)}

        if self._glymphatic is not None:
            try:
                data["modules"]["glymphatic"] = self._glymphatic.export_data()
            except Exception as e:
                data["modules"]["glymphatic"] = {"error": str(e)}

        return data

    def get_active_alerts(self) -> list[str]:
        """Get list of currently active alerts."""
        return [
            alert for alert, count in self._active_alerts.items()
            if count > 0
        ]

    def get_recent_events(self, n: int = 10) -> list[CrossScaleEvent]:
        """Get n most recent cross-scale events."""
        return self._events[-n:]

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def create_unified_dashboard(self, fig=None):
        """
        Create unified multi-scale dashboard.

        Layout:
        - Top: Timeline overview (DA, stability, PAC)
        - Middle: Module-specific panels
        - Bottom: Health and alerts
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if fig is None:
            fig = plt.figure(figsize=(20, 16))

        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # Top row: Unified timeline
        ax_timeline = fig.add_subplot(gs[0, :])
        self._plot_unified_timeline(ax_timeline)

        # Second row: DA | PAC | Stability
        ax_da = fig.add_subplot(gs[1, 0])
        ax_pac = fig.add_subplot(gs[1, 1])
        ax_stab = fig.add_subplot(gs[1, 2])

        if self._da is not None:
            self._da.plot_rpe_distribution(ax=ax_da)
        else:
            ax_da.text(0.5, 0.5, "DA not registered", ha="center", va="center")
            ax_da.set_title("DA Telemetry")

        if self._pac is not None:
            self._pac.plot_comodulogram(ax=ax_pac)
        else:
            ax_pac.text(0.5, 0.5, "PAC not registered", ha="center", va="center")
            ax_pac.set_title("PAC Telemetry")

        if self._stability is not None:
            from t4dm.visualization.stability_monitor import plot_eigenvalue_spectrum
            plot_eigenvalue_spectrum(self._stability, ax=ax_stab)
        else:
            ax_stab.text(0.5, 0.5, "Stability not registered", ha="center", va="center")
            ax_stab.set_title("Stability Monitor")

        # Third row: Cross-scale correlations | Events | Health
        ax_corr = fig.add_subplot(gs[2, 0])
        ax_events = fig.add_subplot(gs[2, 1])
        ax_health = fig.add_subplot(gs[2, 2])

        self._plot_correlations(ax_corr)
        self._plot_events(ax_events)
        self._plot_health(ax_health)

        # Bottom row: Summary statistics
        ax_summary = fig.add_subplot(gs[3, :])
        self._plot_summary(ax_summary)

        fig.suptitle(
            "Multi-Scale Telemetry Hub",
            fontsize=16, fontweight="bold"
        )

        return fig

    def _plot_unified_timeline(self, ax):
        """Plot unified timeline across modules."""
        ax.set_title("Unified Timeline")

        traces = {}
        if self._da is not None and self._da._snapshots:
            t0 = self._da._snapshots[0].timestamp
            times = [(s.timestamp - t0).total_seconds() for s in self._da._snapshots]
            da_levels = [s.da_level for s in self._da._snapshots]
            ax.plot(times, da_levels, "r-", label="DA", alpha=0.7)
            traces["da"] = (times, da_levels)

        if self._pac is not None and self._pac._snapshots:
            t0 = self._pac._snapshots[0].timestamp
            times = [(s.timestamp - t0).total_seconds() for s in self._pac._snapshots]
            mi = [s.modulation_index for s in self._pac._snapshots]
            ax.plot(times, mi, "b-", label="PAC MI", alpha=0.7)
            traces["pac"] = (times, mi)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _plot_correlations(self, ax):
        """Plot cross-scale correlations."""
        correlations = self.compute_cross_scale_correlation()

        if correlations.get("insufficient_data", False):
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title("Cross-Scale Correlations")
            return

        labels = ["DA-PAC", "DA-Stab", "PAC-Stab"]
        values = [
            correlations.get("da_pac", 0.0),
            correlations.get("da_stability", 0.0),
            correlations.get("pac_stability", 0.0),
        ]

        colors = ["green" if v > 0 else "red" for v in values]
        ax.barh(labels, values, color=colors, edgecolor="black")
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Correlation")
        ax.set_title("Cross-Scale Correlations")
        ax.grid(True, alpha=0.3, axis="x")

    def _plot_events(self, ax):
        """Plot recent cross-scale events."""
        ax.axis("off")
        ax.set_title("Recent Cross-Scale Events")

        events = self._events[-5:]
        if not events:
            ax.text(0.5, 0.5, "No events", ha="center", va="center")
            return

        text_lines = []
        for e in events:
            severity_symbol = {"info": "ℹ", "warning": "⚠", "critical": "🚨"}.get(
                e.severity, "•"
            )
            text_lines.append(f"{severity_symbol} {e.event_type}: {e.description[:40]}...")

        ax.text(
            0.05, 0.95, "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top"
        )

    def _plot_health(self, ax):
        """Plot system health gauge."""
        health = self.get_system_health()

        # Simple gauge representation
        ax.axis("off")

        status_color = {
            "healthy": "green",
            "warning": "orange",
            "critical": "red",
        }.get(health.overall_status, "gray")

        # Draw gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, "k-", linewidth=2)

        # Pointer based on overall score
        score = (health.stability_score + health.biological_validity + health.cross_scale_coherence) / 3
        angle = np.pi * (1 - score)
        ax.arrow(0, 0, 0.8 * np.cos(angle), 0.8 * np.sin(angle),
                 head_width=0.1, head_length=0.05, fc=status_color, ec=status_color)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.text(0, -0.15, f"Status: {health.overall_status.upper()}",
                ha="center", fontsize=12, fontweight="bold", color=status_color)
        ax.set_title("System Health")

    def _plot_summary(self, ax):
        """Plot summary statistics."""
        ax.axis("off")

        stats = self.get_summary_statistics()

        text = "Summary Statistics\n" + "=" * 50 + "\n"

        for module, data in stats.get("modules", {}).items():
            if isinstance(data, dict) and "error" not in data:
                text += f"\n{module.upper()}:\n"
                for k, v in list(data.items())[:3]:
                    if isinstance(v, float):
                        text += f"  {k}: {v:.3f}\n"
                    else:
                        text += f"  {k}: {v}\n"

        health = stats.get("health", {})
        text += "\nHEALTH:\n"
        text += f"  Status: {health.get('status', 'unknown')}\n"
        text += f"  Stability: {health.get('stability_score', 0):.2f}\n"
        text += f"  Validity: {health.get('biological_validity', 0):.2f}\n"

        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top"
        )

    def clear_all(self) -> None:
        """Clear all telemetry data."""
        self._events.clear()
        self._active_alerts.clear()
        self._correlation_history.clear()
        self._health_history.clear()

        if self._da is not None:
            self._da.clear()
        # Note: Other modules should have their own clear methods


# Convenience functions
def create_telemetry_hub(
    enable_swr: bool = True,
    enable_pac: bool = True,
    enable_da: bool = True,
    enable_stability: bool = True,
    enable_nt: bool = True,
    enable_ff: bool = True,
    enable_capsule: bool = True,
    enable_glymphatic: bool = True,
) -> TelemetryHub:
    """
    Create configured telemetry hub.

    Args:
        enable_swr: Enable SWR (Sharp-Wave Ripple) telemetry
        enable_pac: Enable PAC (Phase-Amplitude Coupling) telemetry
        enable_da: Enable DA (Dopamine) telemetry
        enable_stability: Enable stability monitoring
        enable_nt: Enable NT (Neurotransmitter) state dashboard
        enable_ff: Enable Forward-Forward network telemetry (Phase 4)
        enable_capsule: Enable Capsule network telemetry (Phase 4)
        enable_glymphatic: Enable Glymphatic system telemetry (Phase 4)

    Returns:
        Configured TelemetryHub instance
    """
    config = TelemetryConfig(
        enable_swr=enable_swr,
        enable_pac=enable_pac,
        enable_da=enable_da,
        enable_stability=enable_stability,
        enable_nt=enable_nt,
        enable_ff=enable_ff,
        enable_capsule=enable_capsule,
        enable_glymphatic=enable_glymphatic,
    )
    return TelemetryHub(config=config)


__all__ = [
    "TelemetryHub",
    "TelemetryConfig",
    "TimeScale",
    "CrossScaleEvent",
    "SystemHealth",
    "create_telemetry_hub",
]
