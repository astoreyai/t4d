"""
Tests for Feature Flags (Phase 9).

Tests the centralized subsystem control system.
"""

import os
from unittest.mock import patch

import pytest

from t4dm.core.feature_flags import (
    DEFAULT_FLAGS,
    FeatureFlag,
    FeatureFlags,
    FlagConfig,
    disable_feature,
    enable_feature,
    get_feature_flags,
    is_feature_enabled,
    reset_feature_flags,
)


class TestFlagConfig:
    """Tests for FlagConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = FlagConfig()
        assert config.enabled is False
        assert config.rollout_percentage == 100.0
        assert config.description == ""

    def test_custom_config(self):
        """Test custom configuration."""
        config = FlagConfig(
            enabled=True,
            rollout_percentage=50.0,
            description="Test flag",
            owner="test-team",
        )
        assert config.enabled is True
        assert config.rollout_percentage == 50.0
        assert config.description == "Test flag"
        assert config.owner == "test-team"

    def test_rollout_clamping(self):
        """Test rollout percentage is clamped to 0-100."""
        config1 = FlagConfig(rollout_percentage=-10.0)
        assert config1.rollout_percentage == 0.0

        config2 = FlagConfig(rollout_percentage=150.0)
        assert config2.rollout_percentage == 100.0


class TestFeatureFlag:
    """Tests for FeatureFlag enum."""

    def test_known_flags(self):
        """Test known feature flags exist."""
        assert FeatureFlag.FF_ENCODER.value == "ff_encoder"
        assert FeatureFlag.CAPSULE_ENCODING.value == "capsule_encoding"
        assert FeatureFlag.LABILITY_WINDOW.value == "lability_window"
        assert FeatureFlag.THREE_FACTOR_LEARNING.value == "three_factor_learning"

    def test_emergency_flags(self):
        """Test emergency flags exist."""
        assert FeatureFlag.READ_ONLY_MODE.value == "read_only_mode"
        assert FeatureFlag.MAINTENANCE_MODE.value == "maintenance_mode"


class TestFeatureFlags:
    """Tests for FeatureFlags manager."""

    @pytest.fixture
    def flags(self):
        """Create fresh feature flags manager."""
        reset_feature_flags()
        return FeatureFlags()

    def test_initialization(self, flags):
        """Test flags initialization."""
        assert len(flags._flags) > 0
        stats = flags.get_stats()
        assert stats["total_flags"] > 0

    def test_is_enabled_default_true(self, flags):
        """Test flag enabled by default."""
        # FF_ENCODER is enabled by default
        assert flags.is_enabled(FeatureFlag.FF_ENCODER) is True

    def test_is_enabled_default_false(self, flags):
        """Test flag disabled by default."""
        # MAINTENANCE_MODE is disabled by default
        assert flags.is_enabled(FeatureFlag.MAINTENANCE_MODE) is False

    def test_set_enabled(self, flags):
        """Test enabling/disabling a flag."""
        flags.set_enabled(FeatureFlag.TELEMETRY, True)
        assert flags.is_enabled(FeatureFlag.TELEMETRY) is True

        flags.set_enabled(FeatureFlag.TELEMETRY, False)
        assert flags.is_enabled(FeatureFlag.TELEMETRY) is False

    def test_set_rollout_percentage(self, flags):
        """Test setting rollout percentage."""
        flags.set_rollout_percentage(FeatureFlag.FF_ENCODER, 50.0)
        config = flags.get_config(FeatureFlag.FF_ENCODER)
        assert config.rollout_percentage == 50.0

    def test_is_enabled_for_deterministic(self, flags):
        """Test rollout is deterministic per user."""
        flags.set_rollout_percentage(FeatureFlag.FF_ENCODER, 50.0)

        # Same user should get same result
        result1 = flags.is_enabled_for(FeatureFlag.FF_ENCODER, user_id="user123")
        result2 = flags.is_enabled_for(FeatureFlag.FF_ENCODER, user_id="user123")
        assert result1 == result2

    def test_is_enabled_for_distribution(self, flags):
        """Test rollout approximates target percentage."""
        flags.set_rollout_percentage(FeatureFlag.FF_ENCODER, 50.0)

        enabled_count = sum(
            1 for i in range(1000)
            if flags.is_enabled_for(FeatureFlag.FF_ENCODER, user_id=f"user{i}")
        )

        # Should be roughly 50% (allow 10% margin)
        assert 400 < enabled_count < 600

    def test_is_enabled_for_100_percent(self, flags):
        """Test 100% rollout is always enabled."""
        flags.set_rollout_percentage(FeatureFlag.FF_ENCODER, 100.0)

        for i in range(100):
            assert flags.is_enabled_for(
                FeatureFlag.FF_ENCODER, user_id=f"user{i}"
            ) is True

    def test_is_enabled_for_0_percent(self, flags):
        """Test 0% rollout is always disabled."""
        flags.set_rollout_percentage(FeatureFlag.FF_ENCODER, 0.0)

        for i in range(100):
            assert flags.is_enabled_for(
                FeatureFlag.FF_ENCODER, user_id=f"user{i}"
            ) is False

    def test_is_enabled_for_disabled_flag(self, flags):
        """Test disabled flag returns False regardless of rollout."""
        flags.set_enabled(FeatureFlag.FF_ENCODER, False)
        flags.set_rollout_percentage(FeatureFlag.FF_ENCODER, 100.0)

        assert flags.is_enabled_for(
            FeatureFlag.FF_ENCODER, user_id="anyuser"
        ) is False

    def test_env_override(self):
        """Test environment variable override."""
        reset_feature_flags()
        with patch.dict(os.environ, {"T4DM_FLAG_TELEMETRY": "true"}):
            flags = FeatureFlags()
            assert flags.is_enabled(FeatureFlag.TELEMETRY) is True

        reset_feature_flags()
        with patch.dict(os.environ, {"T4DM_FLAG_FF_ENCODER": "false"}):
            flags = FeatureFlags()
            assert flags.is_enabled(FeatureFlag.FF_ENCODER) is False

    def test_get_all_flags(self, flags):
        """Test getting all flags."""
        all_flags = flags.get_all_flags()
        assert "ff_encoder" in all_flags
        assert "enabled" in all_flags["ff_encoder"]
        assert "rollout_percentage" in all_flags["ff_encoder"]

    def test_get_config(self, flags):
        """Test getting flag config."""
        config = flags.get_config(FeatureFlag.FF_ENCODER)
        assert config is not None
        assert isinstance(config, FlagConfig)

    def test_usage_tracking(self, flags):
        """Test usage counting."""
        # Check flag multiple times
        for _ in range(5):
            flags.is_enabled(FeatureFlag.FF_ENCODER)

        stats = flags.get_stats()
        assert stats["usage_counts"][FeatureFlag.FF_ENCODER] == 5

    def test_reset_usage_counts(self, flags):
        """Test resetting usage counts."""
        flags.is_enabled(FeatureFlag.FF_ENCODER)
        flags.reset_usage_counts()

        stats = flags.get_stats()
        assert len(stats["usage_counts"]) == 0

    def test_listener(self, flags):
        """Test flag change listener."""
        changes = []

        def listener(flag: FeatureFlag, enabled: bool):
            changes.append((flag, enabled))

        flags.add_listener(listener)
        flags.set_enabled(FeatureFlag.TELEMETRY, True)

        assert len(changes) == 1
        assert changes[0] == (FeatureFlag.TELEMETRY, True)

        flags.remove_listener(listener)
        flags.set_enabled(FeatureFlag.TELEMETRY, False)

        assert len(changes) == 1  # Listener removed, no new changes


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_feature_enabled(self):
        """Test is_feature_enabled function."""
        reset_feature_flags()
        # FF_ENCODER is enabled by default
        assert is_feature_enabled(FeatureFlag.FF_ENCODER) is True
        # MAINTENANCE_MODE is disabled by default
        assert is_feature_enabled(FeatureFlag.MAINTENANCE_MODE) is False

    def test_enable_feature(self):
        """Test enable_feature function."""
        reset_feature_flags()
        enable_feature(FeatureFlag.TELEMETRY)
        assert is_feature_enabled(FeatureFlag.TELEMETRY) is True

    def test_disable_feature(self):
        """Test disable_feature function."""
        reset_feature_flags()
        disable_feature(FeatureFlag.FF_ENCODER)
        assert is_feature_enabled(FeatureFlag.FF_ENCODER) is False


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_feature_flags_singleton(self):
        """Test singleton creation."""
        reset_feature_flags()
        f1 = get_feature_flags()
        f2 = get_feature_flags()
        assert f1 is f2

    def test_reset_feature_flags(self):
        """Test singleton reset."""
        f1 = get_feature_flags()
        reset_feature_flags()
        f2 = get_feature_flags()
        assert f1 is not f2


class TestDefaultFlags:
    """Tests for default flag configurations."""

    def test_learning_flags_enabled(self):
        """Test learning flags are enabled by default."""
        assert DEFAULT_FLAGS[FeatureFlag.FF_ENCODER].enabled is True
        assert DEFAULT_FLAGS[FeatureFlag.FF_RETRIEVAL_SCORING].enabled is True
        assert DEFAULT_FLAGS[FeatureFlag.THREE_FACTOR_LEARNING].enabled is True

    def test_capsule_flags_enabled(self):
        """Test capsule flags are enabled by default."""
        assert DEFAULT_FLAGS[FeatureFlag.CAPSULE_ENCODING].enabled is True
        assert DEFAULT_FLAGS[FeatureFlag.CAPSULE_ROUTING].enabled is True
        assert DEFAULT_FLAGS[FeatureFlag.CAPSULE_POSE_LEARNING].enabled is True

    def test_consolidation_flags_enabled(self):
        """Test consolidation flags are enabled by default."""
        assert DEFAULT_FLAGS[FeatureFlag.LABILITY_WINDOW].enabled is True
        assert DEFAULT_FLAGS[FeatureFlag.GENERATIVE_REPLAY].enabled is True
        assert DEFAULT_FLAGS[FeatureFlag.SWR_PHASE_LOCKING].enabled is True

    def test_emergency_flags_disabled(self):
        """Test emergency flags are disabled by default."""
        assert DEFAULT_FLAGS[FeatureFlag.READ_ONLY_MODE].enabled is False
        assert DEFAULT_FLAGS[FeatureFlag.MAINTENANCE_MODE].enabled is False

    def test_telemetry_disabled(self):
        """Test telemetry is disabled by default."""
        assert DEFAULT_FLAGS[FeatureFlag.TELEMETRY].enabled is False
        assert DEFAULT_FLAGS[FeatureFlag.DISTRIBUTED_TRACING].enabled is False
