"""
Feature Flags for World Weaver.

Phase 9: Centralized subsystem control for gradual rollout and emergency shutoff.

Features:
- Runtime toggle of subsystems without restart
- Gradual rollout percentages
- Environment/config-based defaults
- Telemetry for flag usage

Usage:
    from t4dm.core.feature_flags import get_feature_flags, FeatureFlag

    flags = get_feature_flags()

    if flags.is_enabled(FeatureFlag.FF_ENCODER):
        embedding = ff_encoder.encode(x)
    else:
        embedding = x  # Bypass FF encoder

    # Check with rollout percentage
    if flags.is_enabled_for(FeatureFlag.CAPSULE_ROUTING, user_id="user123"):
        # This user is in the rollout
        use_capsules()
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class FeatureFlag(Enum):
    """
    Known feature flags.

    Naming convention: SUBSYSTEM_FEATURE
    """

    # Learning subsystems (Phase 5)
    FF_ENCODER = "ff_encoder"  # Learnable FF encoding layer
    FF_RETRIEVAL_SCORING = "ff_retrieval_scoring"  # FF-based retrieval ranking

    # Capsule subsystems (Phase 6)
    CAPSULE_ENCODING = "capsule_encoding"  # Capsule representation in storage
    CAPSULE_ROUTING = "capsule_routing"  # Dynamic routing by agreement
    CAPSULE_POSE_LEARNING = "capsule_pose_learning"  # Pose weight updates

    # Consolidation subsystems (Phase 7)
    LABILITY_WINDOW = "lability_window"  # Protein synthesis gate
    GENERATIVE_REPLAY = "generative_replay"  # VAE-based replay
    MULTI_NIGHT_SCHEDULING = "multi_night_scheduling"  # Progressive consolidation
    SWR_PHASE_LOCKING = "swr_phase_locking"  # Sharp-wave ripple gating

    # Three-factor learning (Phase 2)
    THREE_FACTOR_LEARNING = "three_factor_learning"  # Neuromodulated learning
    ELIGIBILITY_TRACES = "eligibility_traces"  # TD(λ) credit assignment
    DOPAMINE_GATING = "dopamine_gating"  # DA surprise gating

    # NCA subsystems
    NCA_NEURAL_FIELD = "nca_neural_field"  # Neural cellular automata
    OSCILLATOR_COUPLING = "oscillator_coupling"  # Theta/gamma oscillations
    NEUROMODULATOR_DYNAMICS = "neuromodulator_dynamics"  # NT state machine

    # API features
    API_RATE_LIMITING = "api_rate_limiting"  # Request rate limits
    API_CACHING = "api_caching"  # Response caching
    API_METRICS = "api_metrics"  # Prometheus metrics

    # Observability
    TELEMETRY = "telemetry"  # Usage telemetry
    DISTRIBUTED_TRACING = "distributed_tracing"  # OpenTelemetry tracing

    # Emergency
    READ_ONLY_MODE = "read_only_mode"  # Block all writes
    MAINTENANCE_MODE = "maintenance_mode"  # Block all requests


@dataclass
class FlagConfig:
    """Configuration for a single feature flag."""

    enabled: bool = False
    rollout_percentage: float = 100.0  # 0-100, percentage of users
    description: str = ""
    owner: str = ""  # Team/person responsible
    expires: str | None = None  # ISO date when flag should be removed

    def __post_init__(self):
        self.rollout_percentage = max(0.0, min(100.0, self.rollout_percentage))


# Default flag configurations
DEFAULT_FLAGS: dict[FeatureFlag, FlagConfig] = {
    # Learning - enabled by default after Phase 5
    FeatureFlag.FF_ENCODER: FlagConfig(
        enabled=True,
        description="Learnable FF encoding layer between embedder and storage",
    ),
    FeatureFlag.FF_RETRIEVAL_SCORING: FlagConfig(
        enabled=True,
        description="Use FF goodness for retrieval ranking",
    ),

    # Capsules - enabled by default after Phase 6
    FeatureFlag.CAPSULE_ENCODING: FlagConfig(
        enabled=True,
        description="Store capsule activations with episodes",
    ),
    FeatureFlag.CAPSULE_ROUTING: FlagConfig(
        enabled=True,
        description="Use dynamic routing by agreement",
    ),
    FeatureFlag.CAPSULE_POSE_LEARNING: FlagConfig(
        enabled=True,
        description="Update pose weights from routing agreement",
    ),

    # Consolidation - enabled by default after Phase 7
    FeatureFlag.LABILITY_WINDOW: FlagConfig(
        enabled=True,
        description="6-hour protein synthesis gate for reconsolidation",
    ),
    FeatureFlag.GENERATIVE_REPLAY: FlagConfig(
        enabled=True,
        description="VAE-based synthetic memory generation during sleep",
    ),
    FeatureFlag.MULTI_NIGHT_SCHEDULING: FlagConfig(
        enabled=True,
        description="Progressive consolidation depth across nights",
    ),
    FeatureFlag.SWR_PHASE_LOCKING: FlagConfig(
        enabled=True,
        description="Gate replay by sharp-wave ripple timing",
    ),

    # Three-factor learning
    FeatureFlag.THREE_FACTOR_LEARNING: FlagConfig(
        enabled=True,
        description="Neuromodulator-gated learning rule",
    ),
    FeatureFlag.ELIGIBILITY_TRACES: FlagConfig(
        enabled=True,
        description="TD(λ) eligibility traces for credit assignment",
    ),
    FeatureFlag.DOPAMINE_GATING: FlagConfig(
        enabled=True,
        description="Dopamine prediction error gating",
    ),

    # NCA
    FeatureFlag.NCA_NEURAL_FIELD: FlagConfig(
        enabled=True,
        description="Neural cellular automata field dynamics",
    ),
    FeatureFlag.OSCILLATOR_COUPLING: FlagConfig(
        enabled=True,
        description="Theta/gamma oscillator coupling",
    ),
    FeatureFlag.NEUROMODULATOR_DYNAMICS: FlagConfig(
        enabled=True,
        description="Neuromodulator state machine",
    ),

    # API features
    FeatureFlag.API_RATE_LIMITING: FlagConfig(
        enabled=True,
        description="Request rate limiting",
    ),
    FeatureFlag.API_CACHING: FlagConfig(
        enabled=False,
        description="Response caching (disabled by default)",
    ),
    FeatureFlag.API_METRICS: FlagConfig(
        enabled=True,
        description="Prometheus metrics endpoint",
    ),

    # Observability
    FeatureFlag.TELEMETRY: FlagConfig(
        enabled=False,
        description="Usage telemetry collection",
    ),
    FeatureFlag.DISTRIBUTED_TRACING: FlagConfig(
        enabled=False,
        description="OpenTelemetry distributed tracing",
    ),

    # Emergency - always start disabled
    FeatureFlag.READ_ONLY_MODE: FlagConfig(
        enabled=False,
        description="Block all write operations",
    ),
    FeatureFlag.MAINTENANCE_MODE: FlagConfig(
        enabled=False,
        description="Block all API requests",
    ),
}


class FeatureFlags:
    """
    Centralized feature flag manager.

    Provides runtime control over World Weaver subsystems with:
    - Environment variable overrides
    - Percentage-based rollout
    - Runtime updates without restart
    - Usage telemetry

    Example:
        flags = FeatureFlags()

        # Simple check
        if flags.is_enabled(FeatureFlag.FF_ENCODER):
            use_ff_encoder()

        # Rollout check (deterministic per user)
        if flags.is_enabled_for(FeatureFlag.NEW_FEATURE, user_id="u123"):
            use_new_feature()

        # Runtime toggle
        flags.set_enabled(FeatureFlag.READ_ONLY_MODE, True)
    """

    def __init__(
        self,
        defaults: dict[FeatureFlag, FlagConfig] | None = None,
        env_prefix: str = "T4DM_FLAG_",
    ):
        """
        Initialize feature flags.

        Args:
            defaults: Default flag configurations
            env_prefix: Environment variable prefix for overrides
        """
        self._lock = threading.RLock()
        self._flags: dict[FeatureFlag, FlagConfig] = {}
        self._env_prefix = env_prefix
        self._usage_counts: dict[FeatureFlag, int] = {}
        self._listeners: list[Callable[[FeatureFlag, bool], None]] = []

        # Initialize from defaults
        defaults = defaults or DEFAULT_FLAGS
        for flag, config in defaults.items():
            self._flags[flag] = FlagConfig(
                enabled=config.enabled,
                rollout_percentage=config.rollout_percentage,
                description=config.description,
                owner=config.owner,
                expires=config.expires,
            )

        # Apply environment overrides
        self._apply_env_overrides()

        logger.info(f"FeatureFlags initialized with {len(self._flags)} flags")

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        for flag in FeatureFlag:
            env_key = f"{self._env_prefix}{flag.value.upper()}"
            env_value = os.environ.get(env_key)

            if env_value is not None:
                enabled = env_value.lower() in ("true", "1", "yes", "on")
                if flag in self._flags:
                    self._flags[flag].enabled = enabled
                else:
                    self._flags[flag] = FlagConfig(enabled=enabled)

                logger.debug(f"Flag {flag.value} overridden to {enabled} via env")

    def is_enabled(self, flag: FeatureFlag) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag: Feature flag to check

        Returns:
            True if enabled
        """
        with self._lock:
            config = self._flags.get(flag)
            if config is None:
                return False

            # Track usage
            self._usage_counts[flag] = self._usage_counts.get(flag, 0) + 1

            return config.enabled

    def is_enabled_for(
        self,
        flag: FeatureFlag,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """
        Check if a feature is enabled for a specific user/session.

        Uses rollout percentage for gradual rollout.
        Deterministic: same user always gets same result.

        Args:
            flag: Feature flag to check
            user_id: User identifier for rollout
            session_id: Session identifier (fallback if no user_id)

        Returns:
            True if enabled for this user
        """
        with self._lock:
            config = self._flags.get(flag)
            if config is None:
                return False

            if not config.enabled:
                return False

            # If 100% rollout, always enabled
            if config.rollout_percentage >= 100.0:
                return True

            # If 0% rollout, always disabled
            if config.rollout_percentage <= 0.0:
                return False

            # Compute deterministic hash for user
            identifier = user_id or session_id or "default"
            hash_input = f"{flag.value}:{identifier}"
            hash_bytes = hashlib.md5(hash_input.encode(), usedforsecurity=False).digest()
            hash_value = int.from_bytes(hash_bytes[:4], "little")
            percentage = (hash_value % 10000) / 100.0  # 0.00 to 99.99

            return percentage < config.rollout_percentage

    def set_enabled(self, flag: FeatureFlag, enabled: bool) -> None:
        """
        Enable or disable a feature flag at runtime.

        Args:
            flag: Feature flag to modify
            enabled: New enabled state
        """
        with self._lock:
            if flag not in self._flags:
                self._flags[flag] = FlagConfig()

            old_enabled = self._flags[flag].enabled
            self._flags[flag].enabled = enabled

            logger.info(f"Flag {flag.value} changed: {old_enabled} -> {enabled}")

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(flag, enabled)
                except Exception as e:
                    logger.warning(f"Flag listener error: {e}")

    def set_rollout_percentage(self, flag: FeatureFlag, percentage: float) -> None:
        """
        Set the rollout percentage for a flag.

        Args:
            flag: Feature flag to modify
            percentage: Rollout percentage (0-100)
        """
        with self._lock:
            if flag not in self._flags:
                self._flags[flag] = FlagConfig()

            old_pct = self._flags[flag].rollout_percentage
            self._flags[flag].rollout_percentage = max(0.0, min(100.0, percentage))

            logger.info(
                f"Flag {flag.value} rollout: {old_pct}% -> "
                f"{self._flags[flag].rollout_percentage}%"
            )

    def get_config(self, flag: FeatureFlag) -> FlagConfig | None:
        """Get the configuration for a flag."""
        with self._lock:
            return self._flags.get(flag)

    def get_all_flags(self) -> dict[str, dict]:
        """Get all flag states."""
        with self._lock:
            return {
                flag.value: {
                    "enabled": config.enabled,
                    "rollout_percentage": config.rollout_percentage,
                    "description": config.description,
                    "owner": config.owner,
                    "expires": config.expires,
                    "usage_count": self._usage_counts.get(flag, 0),
                }
                for flag, config in self._flags.items()
            }

    def add_listener(self, listener: Callable[[FeatureFlag, bool], None]) -> None:
        """Add a listener for flag changes."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[FeatureFlag, bool], None]) -> None:
        """Remove a flag change listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def get_stats(self) -> dict:
        """Get feature flag statistics."""
        with self._lock:
            enabled_count = sum(
                1 for c in self._flags.values() if c.enabled
            )
            return {
                "total_flags": len(self._flags),
                "enabled_flags": enabled_count,
                "disabled_flags": len(self._flags) - enabled_count,
                "usage_counts": dict(self._usage_counts),
                "listeners": len(self._listeners),
            }

    def reset_usage_counts(self) -> None:
        """Reset usage counters."""
        with self._lock:
            self._usage_counts.clear()


# ============================================================================
# Convenience Functions
# ============================================================================


def is_feature_enabled(flag: FeatureFlag) -> bool:
    """Quick check if a feature is enabled."""
    return get_feature_flags().is_enabled(flag)


def enable_feature(flag: FeatureFlag) -> None:
    """Enable a feature flag."""
    get_feature_flags().set_enabled(flag, True)


def disable_feature(flag: FeatureFlag) -> None:
    """Disable a feature flag."""
    get_feature_flags().set_enabled(flag, False)


# ============================================================================
# Singleton
# ============================================================================

_feature_flags: FeatureFlags | None = None
_lock = threading.Lock()


def get_feature_flags() -> FeatureFlags:
    """Get or create the singleton feature flags manager."""
    global _feature_flags
    if _feature_flags is None:
        with _lock:
            if _feature_flags is None:
                _feature_flags = FeatureFlags()
    return _feature_flags


def reset_feature_flags() -> None:
    """Reset singleton (for testing)."""
    global _feature_flags
    with _lock:
        _feature_flags = None


__all__ = [
    "DEFAULT_FLAGS",
    "FeatureFlag",
    "FeatureFlags",
    "FlagConfig",
    "disable_feature",
    "enable_feature",
    "get_feature_flags",
    "is_feature_enabled",
    "reset_feature_flags",
]
