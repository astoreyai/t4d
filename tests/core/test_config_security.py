"""Tests for configuration security validation."""
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ww.core.config import Settings, validate_password_strength


class TestPasswordValidation:
    """Test password validation logic."""

    def test_empty_password_rejected(self):
        """Empty password should be rejected."""
        # Disable test mode to enable password validation
        with patch.dict(os.environ, {"WW_NEO4J_PASSWORD": "", "WW_TEST_MODE": "false"}, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "required" in str(exc_info.value).lower()

    def test_short_password_rejected(self):
        """Password < 12 chars should be rejected (P2-SEC-M1)."""
        # Disable test mode to enable password validation
        with patch.dict(os.environ, {"WW_NEO4J_PASSWORD": "Short123!", "WW_TEST_MODE": "false"}, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "12 characters" in str(exc_info.value)

    def test_weak_password_rejected(self):
        """Common weak passwords should be rejected."""
        # Use passwords from the WEAK_PASSWORDS list (some need 12+ chars to test weak detection)
        weak_passwords = ["passwordpassword", "admin123admin", "changemechange"]

        for weak in weak_passwords:
            # Disable test mode to enable password validation
            with patch.dict(os.environ, {"WW_NEO4J_PASSWORD": weak, "WW_TEST_MODE": "false"}, clear=False):
                with pytest.raises(ValidationError) as exc_info:
                    Settings()
                error_msg = str(exc_info.value).lower()
                # Should be rejected for being weak OR lacking complexity OR too short
                assert "weak" in error_msg or "complexity" in error_msg or "12 characters" in error_msg, \
                    f"Password '{weak}' should be rejected but got: {exc_info.value}"

    def test_strong_password_accepted(self):
        """Strong passwords should be accepted (P2-SEC-M1: 12+ chars, 3/4 classes)."""
        strong_passwords = [
            "MyS3cur3P@ss!word",  # upper + lower + digit + special (17 chars)
            "correct-horse-Battery1",  # lower + special + upper + digit (22 chars)
            "Tr0ub4dor&3More!",  # upper + lower + digit + special (16 chars)
            "xK9#mP2$vL5@nQ8Y",  # upper + lower + digit + special (16 chars)
        ]

        for strong in strong_passwords:
            with patch.dict(os.environ, {"WW_NEO4J_PASSWORD": strong}, clear=False):
                # Should not raise
                settings = Settings()
                assert settings.neo4j_password == strong

    def test_test_mode_allows_empty(self):
        """Test mode should allow empty/weak passwords."""
        with patch.dict(os.environ, {
            "WW_TEST_MODE": "true",
            "WW_NEO4J_PASSWORD": "",
        }, clear=False):
            settings = Settings()
            assert settings.neo4j_password == "test-password"

    def test_test_environment_allows_empty(self):
        """Test environment should allow empty passwords."""
        with patch.dict(os.environ, {
            "WW_ENVIRONMENT": "test",
            "WW_NEO4J_PASSWORD": "",
        }, clear=False):
            settings = Settings()
            assert settings.neo4j_password == "test-password"


class TestProductionValidation:
    """Test production environment validation."""

    def test_production_warns_missing_qdrant_key(self, caplog):
        """Production should warn about missing Qdrant API key."""
        with patch.dict(os.environ, {
            "WW_ENVIRONMENT": "production",
            "WW_NEO4J_PASSWORD": "SecureP@ss123!x",  # 15 chars, 4 classes
            "WW_QDRANT_API_KEY": "",
        }, clear=False):
            Settings()
            assert "QDRANT_API_KEY" in caplog.text or "qdrant" in caplog.text.lower()

    def test_production_warns_insecure_otel(self, caplog):
        """Production should warn about insecure OTEL."""
        with patch.dict(os.environ, {
            "WW_ENVIRONMENT": "production",
            "WW_NEO4J_PASSWORD": "SecureP@ss123!x",  # 15 chars, 4 classes
            "WW_OTEL_INSECURE": "true",
        }, clear=False):
            Settings()
            assert "insecure" in caplog.text.lower()

    def test_production_multiple_warnings(self, caplog):
        """Production should warn about multiple security issues."""
        with patch.dict(os.environ, {
            "WW_ENVIRONMENT": "production",
            "WW_NEO4J_PASSWORD": "SecureP@ss123!x",  # 15 chars, 4 classes
            "WW_OTEL_INSECURE": "true",
            "WW_QDRANT_API_KEY": "",
        }, clear=False):
            Settings()
            log_text = caplog.text.lower()
            assert "insecure" in log_text
            # Production warnings should be present


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
            validate_password_strength("Short123!", "test")  # 9 chars with complexity
        assert "12 characters" in str(exc_info.value)

    def test_weak_common_password_fails(self):
        """Common weak passwords should fail."""
        # Use passwords that would otherwise meet requirements but are in the weak list
        weak_passwords = ["passwordpassword", "admin123admin123", "changemechange"]

        for weak in weak_passwords:
            with pytest.raises(ValueError) as exc_info:
                validate_password_strength(weak, "test")
            error_msg = str(exc_info.value).lower()
            # Should be rejected for being weak OR lacking complexity OR being too short
            assert "weak" in error_msg or "complexity" in error_msg or "12 characters" in error_msg

    def test_complexity_check(self):
        """Password needs 3+ character types (P2-SEC-M1)."""
        # Only lowercase (12 chars) - should fail for complexity
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("alllowercase", "test")
        assert "complexity" in str(exc_info.value).lower()

        # Only 2 types (lower + digit, 12 chars) - should fail for complexity
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("lower123case", "test")
        assert "complexity" in str(exc_info.value).lower()

        # Only 2 types (upper + special, 12 chars) - should fail for complexity
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("UPPER@@@@@@@", "test")
        assert "complexity" in str(exc_info.value).lower()

        # Only 2 types (lower + upper, 12 chars) - should fail for complexity
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("LowerUpperAA", "test")
        assert "complexity" in str(exc_info.value).lower()

        # 3 types (lower + upper + digit, 12 chars) - should pass
        # Note: Must not match weak patterns like ^[a-z]+\d+$ (case-insensitive)
        result = validate_password_strength("Lower12Upper", "test")
        assert result == "Lower12Upper"

        # 3 types (lower + digit + special, 12 chars) - should pass
        result = validate_password_strength("lower123@@@@", "test")
        assert result == "lower123@@@@"

        # 4 types (all, 12 chars) - should pass
        result = validate_password_strength("Lower123@Up!", "test")
        assert result == "Lower123@Up!"

    def test_field_name_in_error(self):
        """Error messages should include field name."""
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("", "my_field")
        assert "my_field" in str(exc_info.value)

    def test_strong_passwords_accepted(self):
        """Various strong passwords should be accepted (P2-SEC-M1: 12+ chars, 3/4 classes)."""
        strong_passwords = [
            "MyPassword123!",  # upper + lower + digit + special (14 chars)
            "Secure@pass123",  # upper + lower + digit + special (14 chars)
            "P@ssw0rd!Extra",  # upper + lower + digit + special (14 chars)
            "correct-Horse-1",  # lower + special + upper + digit (15 chars)
            "Tr0ub4dor&3More",  # upper + lower + digit + special (15 chars)
        ]

        for strong in strong_passwords:
            result = validate_password_strength(strong, "test")
            assert result == strong

    def test_weak_patterns_rejected(self):
        """P2-SEC-M1: Passwords matching weak patterns should be rejected."""
        # Passwords that match WEAK_PATTERNS even if 12+ chars
        pattern_passwords = [
            "password123456",  # matches ^password\d*$
            "welcome1234567",  # matches ^welcome\d*$
            "admin123456789",  # matches ^admin\d*$
            "qwerty12345678",  # matches ^qwerty\d*$
            "letmein1234567",  # matches ^letmein\d*$
            "123456789012",    # matches ^\d+$
            "test12345678901", # matches ^test\d*$
            "user12345678901", # matches ^user\d*$
            "guest1234567890", # matches ^guest\d*$
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
            with patch.dict(os.environ, {
                "WW_ENVIRONMENT": env,
                "WW_NEO4J_PASSWORD": "SecureP@ss123!x",  # 15 chars, 4 classes
            }, clear=False):
                settings = Settings()
                assert settings.environment == env.lower()

    def test_invalid_environment_rejected(self):
        """Invalid environment values should be rejected."""
        with patch.dict(os.environ, {
            "WW_ENVIRONMENT": "invalid",
            "WW_NEO4J_PASSWORD": "SecureP@ss123!x",  # 15 chars, 4 classes
        }, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "environment" in str(exc_info.value).lower()

    def test_environment_case_insensitive(self):
        """Environment validation should be case-insensitive."""
        with patch.dict(os.environ, {
            "WW_ENVIRONMENT": "PRODUCTION",
            "WW_NEO4J_PASSWORD": "SecureP@ss123!x",  # 15 chars, 4 classes
        }, clear=False):
            settings = Settings()
            assert settings.environment == "production"


class TestLogSafeConfig:
    """Test safe logging of configuration."""

    def test_log_safe_config_masks_password(self):
        """log_safe_config should mask passwords."""
        with patch.dict(os.environ, {
            "WW_NEO4J_PASSWORD": "MySecretP@ss123!x",  # 17 chars, 4 classes
        }, clear=False):
            settings = Settings()
            safe_config = settings.log_safe_config()

            # Password should be masked
            assert safe_config["neo4j_password"] != "MySecretP@ss123!x"
            assert "*" in safe_config["neo4j_password"]

    def test_log_safe_config_masks_qdrant_key(self):
        """log_safe_config should mask Qdrant API key."""
        with patch.dict(os.environ, {
            "WW_NEO4J_PASSWORD": "MySecretP@ss123!x",  # 17 chars, 4 classes
            "WW_QDRANT_API_KEY": "my-secret-api-key-12345",
        }, clear=False):
            settings = Settings()
            safe_config = settings.log_safe_config()

            # API key should be masked
            assert safe_config["qdrant_api_key"] != "my-secret-api-key-12345"
            assert "*" in safe_config["qdrant_api_key"]

    def test_log_safe_config_preserves_non_secrets(self):
        """log_safe_config should preserve non-secret fields."""
        with patch.dict(os.environ, {
            "WW_NEO4J_PASSWORD": "MySecretP@ss123!x",  # 17 chars, 4 classes
            "WW_NEO4J_URI": "bolt://custom-host:7687",
            "WW_SESSION_ID": "test-session",
        }, clear=False):
            settings = Settings()
            safe_config = settings.log_safe_config()

            # Non-secret fields should be preserved
            assert safe_config["neo4j_uri"] == "bolt://custom-host:7687"
            assert safe_config["session_id"] == "test-session"
