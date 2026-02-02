"""Tests for configuration security validation."""
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from t4dm.core.config import Settings, validate_password_strength


class TestPasswordValidation:
    """Test password validation logic (Neo4j removed — validator is standalone)."""

    def test_validate_password_strength_exists(self):
        """validate_password_strength function is importable."""
        assert callable(validate_password_strength)


class TestProductionValidation:
    """Test production environment validation."""

    def test_production_warns_insecure_otel(self, caplog):
        """Production should warn about insecure OTEL."""
        with patch.dict(os.environ, {
            "T4DM_ENVIRONMENT": "production",
            "T4DM_OTEL_INSECURE": "true",
        }, clear=False):
            Settings()
            assert "insecure" in caplog.text.lower()


class TestPasswordStrengthValidator:
    """Test the validate_password_strength function directly."""

    def test_empty_password_fails(self):
        """Empty password should fail."""
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("", "test")
        assert "required" in str(exc_info.value).lower()

    def test_short_password_fails(self):
        """Password < 12 chars should fail (P2-SEC-M1)."""
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("Short123!", "test")
        assert "12 characters" in str(exc_info.value)

    def test_weak_common_password_fails(self):
        """Common weak passwords should fail."""
        weak_passwords = ["passwordpassword", "admin123admin123", "changemechange"]
        for weak in weak_passwords:
            with pytest.raises(ValueError) as exc_info:
                validate_password_strength(weak, "test")
            error_msg = str(exc_info.value).lower()
            assert "weak" in error_msg or "complexity" in error_msg or "12 characters" in error_msg

    def test_complexity_check(self):
        """Password needs 3+ character types (P2-SEC-M1)."""
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("alllowercase", "test")
        assert "complexity" in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("lower123case", "test")
        assert "complexity" in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("UPPER@@@@@@@", "test")
        assert "complexity" in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("LowerUpperAA", "test")
        assert "complexity" in str(exc_info.value).lower()

        result = validate_password_strength("Lower12Upper", "test")
        assert result == "Lower12Upper"

        result = validate_password_strength("lower123@@@@", "test")
        assert result == "lower123@@@@"

        result = validate_password_strength("Lower123@Up!", "test")
        assert result == "Lower123@Up!"

    def test_field_name_in_error(self):
        """Error messages should include field name."""
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("", "my_field")
        assert "my_field" in str(exc_info.value)

    def test_strong_passwords_accepted(self):
        """Various strong passwords should be accepted."""
        strong_passwords = [
            "MyPassword123!",
            "Secure@pass123",
            "P@ssw0rd!Extra",
            "correct-Horse-1",
            "Tr0ub4dor&3More",
        ]
        for strong in strong_passwords:
            result = validate_password_strength(strong, "test")
            assert result == strong

    def test_weak_patterns_rejected(self):
        """P2-SEC-M1: Passwords matching weak patterns should be rejected."""
        pattern_passwords = [
            "password123456",
            "welcome1234567",
            "admin123456789",
            "qwerty12345678",
            "letmein1234567",
            "123456789012",
            "test12345678901",
            "user12345678901",
            "guest1234567890",
        ]
        for weak in pattern_passwords:
            with pytest.raises(ValueError) as exc_info:
                validate_password_strength(weak, "test")
            error_msg = str(exc_info.value).lower()
            assert "weak" in error_msg or "complexity" in error_msg, \
                f"Password '{weak}' should be rejected but got: {exc_info.value}"


class TestEnvironmentValidation:
    """Test environment field validation."""

    def test_valid_environments_accepted(self):
        """Valid environment values should be accepted."""
        valid_envs = ["development", "staging", "production", "test"]
        for env in valid_envs:
            with patch.dict(os.environ, {"T4DM_ENVIRONMENT": env}, clear=False):
                settings = Settings()
                assert settings.environment == env.lower()

    def test_invalid_environment_rejected(self):
        """Invalid environment values should be rejected."""
        with patch.dict(os.environ, {"T4DM_ENVIRONMENT": "invalid"}, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "environment" in str(exc_info.value).lower()

    def test_environment_case_insensitive(self):
        """Environment validation should be case-insensitive."""
        with patch.dict(os.environ, {"T4DM_ENVIRONMENT": "PRODUCTION"}, clear=False):
            settings = Settings()
            assert settings.environment == "production"


class TestLogSafeConfig:
    """Test safe logging of configuration."""

    def test_log_safe_config_masks_api_key(self):
        """log_safe_config should mask API keys."""
        with patch.dict(os.environ, {"T4DM_API_KEY": "my-secret-api-key-12345"}, clear=False):
            settings = Settings()
            safe_config = settings.log_safe_config()
            assert safe_config["api_key"] != "my-secret-api-key-12345"
            assert "*" in safe_config["api_key"]

    def test_log_safe_config_masks_admin_api_key(self):
        """log_safe_config should mask admin API key."""
        with patch.dict(os.environ, {"T4DM_ADMIN_API_KEY": "my-admin-secret-key"}, clear=False):
            settings = Settings()
            safe_config = settings.log_safe_config()
            assert safe_config["admin_api_key"] != "my-admin-secret-key"
            assert "*" in safe_config["admin_api_key"]

    def test_log_safe_config_preserves_non_secrets(self):
        """log_safe_config should preserve non-secret fields."""
        with patch.dict(os.environ, {"T4DM_SESSION_ID": "test-session"}, clear=False):
            settings = Settings()
            safe_config = settings.log_safe_config()
            assert safe_config["session_id"] == "test-session"
