"""
Unit tests for configuration security features.

Tests masking, permission validation, and weight validation.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from t4dm.core.config import (
    Settings,
    check_file_permissions,
    load_secret_from_env,
    mask_secret,
    validate_weights,
)


class TestMaskSecret:
    """Test secret masking function."""

    def test_mask_simple_password(self):
        """Test masking a simple password."""
        result = mask_secret("password123")
        assert result == "pass*******"

    def test_mask_short_value(self):
        """Test masking a value shorter than visible_chars."""
        result = mask_secret("abc")
        assert result == "***"

    def test_mask_empty_string(self):
        """Test masking empty string."""
        result = mask_secret("")
        assert result == "***"

    def test_mask_custom_visible_chars(self):
        """Test masking with custom visible character count."""
        result = mask_secret("secretkey12345", visible_chars=6)
        assert result == "secret********"

    def test_mask_api_key(self):
        """Test masking API key format."""
        api_key = "sk-1234567890abcdef"
        result = mask_secret(api_key)
        # sk-1234567890abcdef is 18 chars, first 4 shown = 14 asterisks
        assert result == "sk-1***************"


class TestCheckFilePermissions:
    """Test file permission checking."""

    def test_nonexistent_file(self, caplog):
        """Test that nonexistent file doesn't raise error."""
        with caplog.at_level(logging.WARNING):
            check_file_permissions(Path("/nonexistent/file.txt"))
        assert len(caplog.records) == 0

    def test_secure_permissions(self, caplog, tmp_path):
        """Test that secure file permissions don't trigger warning."""
        test_file = tmp_path / "secure.env"
        test_file.write_text("SECRET=value")
        test_file.chmod(0o600)  # Owner read/write only

        with caplog.at_level(logging.WARNING):
            check_file_permissions(test_file)
        assert len(caplog.records) == 0

    def test_group_readable_warns(self, caplog, tmp_path):
        """Test that group-readable file triggers warning."""
        test_file = tmp_path / "group_readable.env"
        test_file.write_text("SECRET=value")
        test_file.chmod(0o640)  # Owner read/write, group read

        with caplog.at_level(logging.WARNING):
            check_file_permissions(test_file)

        assert len(caplog.records) == 1
        assert "permissive permissions" in caplog.records[0].message
        assert str(test_file) in caplog.records[0].message
        assert "chmod 600" in caplog.records[0].message

    def test_world_readable_warns(self, caplog, tmp_path):
        """Test that world-readable file triggers warning."""
        test_file = tmp_path / "world_readable.env"
        test_file.write_text("SECRET=value")
        test_file.chmod(0o644)  # Owner read/write, group/others read

        with caplog.at_level(logging.WARNING):
            check_file_permissions(test_file)

        assert len(caplog.records) == 1
        assert "permissive permissions" in caplog.records[0].message

    def test_enforce_raises_error(self, tmp_path):
        """P3-SEC-L1: Test that enforce=True raises PermissionError."""
        test_file = tmp_path / "insecure.env"
        test_file.write_text("SECRET=value")
        test_file.chmod(0o644)

        with pytest.raises(PermissionError, match="permissive permissions"):
            check_file_permissions(test_file, enforce=True)

    def test_enforce_secure_file_ok(self, tmp_path):
        """P3-SEC-L1: Test that enforce=True doesn't raise for secure file."""
        test_file = tmp_path / "secure.env"
        test_file.write_text("SECRET=value")
        test_file.chmod(0o600)

        # Should not raise
        check_file_permissions(test_file, enforce=True)

    def test_auto_fix_permissions(self, tmp_path, caplog):
        """P3-SEC-L1: Test that auto_fix=True fixes permissions."""
        test_file = tmp_path / "fixable.env"
        test_file.write_text("SECRET=value")
        test_file.chmod(0o644)

        with caplog.at_level(logging.INFO):
            check_file_permissions(test_file, auto_fix=True)

        # Permissions should be fixed
        assert (test_file.stat().st_mode & 0o777) == 0o600
        assert "Auto-fixed" in caplog.text

    def test_auto_fix_with_enforce(self, tmp_path):
        """P3-SEC-L1: Test auto_fix with enforce doesn't raise after fix."""
        test_file = tmp_path / "fixable.env"
        test_file.write_text("SECRET=value")
        test_file.chmod(0o644)

        # Should fix and not raise
        check_file_permissions(test_file, enforce=True, auto_fix=True)

        # Permissions should be fixed
        assert (test_file.stat().st_mode & 0o777) == 0o600


class TestLoadSecretFromEnv:
    """Test environment variable loading with masking."""

    def test_load_existing_var(self, caplog):
        """Test loading existing environment variable."""
        with patch.dict(os.environ, {"TEST_SECRET": "mypassword123"}):
            with caplog.at_level(logging.DEBUG):
                result = load_secret_from_env("TEST_SECRET")

            assert result == "mypassword123"
            assert len(caplog.records) == 1
            assert "TEST_SECRET" in caplog.records[0].message
            # mypassword123 is 13 chars, first 4 shown = 9 asterisks
            assert "mypa*********" in caplog.records[0].message

    def test_load_missing_var_with_default(self, caplog):
        """Test loading missing variable with default."""
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.DEBUG):
                result = load_secret_from_env("MISSING_VAR", default="default_value")

            assert result == "default_value"

    def test_load_missing_var_no_default(self, caplog):
        """Test loading missing variable without default."""
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.DEBUG):
                result = load_secret_from_env("MISSING_VAR")

            assert result is None
            assert len(caplog.records) == 1
            assert "not set" in caplog.records[0].message


class TestValidateWeights:
    """Test weight validation function."""

    def test_valid_weights(self):
        """Test valid weights that sum to 1.0."""
        weights = {
            "semantic": 0.4,
            "recency": 0.25,
            "outcome": 0.2,
            "importance": 0.15,
        }
        result = validate_weights(weights)
        assert result == weights

    def test_weights_with_tolerance(self):
        """Test weights within tolerance."""
        weights = {
            "semantic": 0.4001,
            "recency": 0.2499,
            "outcome": 0.2,
            "importance": 0.15,
        }
        result = validate_weights(weights)
        assert result == weights

    def test_empty_weights(self):
        """Test empty weights dict."""
        result = validate_weights({})
        assert result == {}

    def test_weights_exceeds_tolerance(self):
        """Test weights that exceed tolerance."""
        weights = {
            "semantic": 0.5,
            "recency": 0.3,
            "outcome": 0.2,
            "importance": 0.1,
        }
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_weights(weights)

    def test_weight_below_zero(self):
        """Test weight below 0."""
        weights = {
            "semantic": -0.1,
            "recency": 0.5,
            "outcome": 0.3,
            "importance": 0.3,
        }
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            validate_weights(weights)

    def test_weight_above_one(self):
        """Test weight above 1."""
        weights = {
            "semantic": 1.5,
            "recency": 0.0,
            "outcome": 0.0,
            "importance": -0.5,
        }
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            validate_weights(weights)

    def test_non_numeric_weight(self):
        """Test non-numeric weight."""
        weights = {
            "semantic": "invalid",
            "recency": 0.5,
            "outcome": 0.3,
            "importance": 0.2,
        }
        with pytest.raises(ValueError, match="must be numeric"):
            validate_weights(weights)


class TestSettingsValidation:
    """Test Settings class validation."""

    def test_default_weights_valid(self):
        """Test that default weights are valid."""
        with patch.dict(os.environ, {"T4DM_TEST_MODE": "true"}):
            settings = Settings()
            assert settings.retrieval_semantic_weight == 0.4
            assert settings.retrieval_recency_weight == 0.25
            assert settings.retrieval_outcome_weight == 0.2
            assert settings.retrieval_importance_weight == 0.15

    def test_custom_valid_weights(self):
        """Test custom valid weights."""
        settings = Settings(
            retrieval_semantic_weight=0.5,
            retrieval_recency_weight=0.3,
            retrieval_outcome_weight=0.1,
            retrieval_importance_weight=0.1,
        )
        assert settings.retrieval_semantic_weight == 0.5

    def test_invalid_weight_sum(self):
        """Test that invalid weight sum raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            Settings(
                episodic_weight_semantic=0.5,
                episodic_weight_recency=0.3,
                episodic_weight_outcome=0.1,
                episodic_weight_importance=0.05,  # Sum = 0.95, not 1.0
            )

    def test_negative_weight(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValidationError):
            Settings(
                episodic_weight_semantic=-0.1,
                episodic_weight_recency=0.5,
                episodic_weight_outcome=0.3,
                episodic_weight_importance=0.3,
            )

    def test_weight_above_one(self):
        """Test that weight above 1 raises error."""
        with pytest.raises(ValidationError):
            Settings(
                episodic_weight_semantic=1.5,
                episodic_weight_recency=0.0,
                episodic_weight_outcome=0.0,
                episodic_weight_importance=-0.5,
            )


class TestSettingsSecurityMethods:
    """Test Settings security methods."""

    def test_validate_permissions_nonexistent(self, caplog):
        """Test permission validation with nonexistent file."""
        with caplog.at_level(logging.WARNING):
            Settings.validate_permissions(Path("/nonexistent/.env"))
        assert len(caplog.records) == 0

    def test_validate_permissions_secure_file(self, caplog, tmp_path):
        """Test permission validation with secure file."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value")
        env_file.chmod(0o600)

        with caplog.at_level(logging.WARNING):
            Settings.validate_permissions(env_file)
        assert len(caplog.records) == 0

    def test_validate_permissions_insecure_file(self, caplog, tmp_path):
        """Test permission validation with insecure file."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value")
        env_file.chmod(0o644)

        with caplog.at_level(logging.WARNING):
            Settings.validate_permissions(env_file)
        assert len(caplog.records) == 1

    def test_load_with_masking(self, caplog):
        """Test loading field with masking."""
        settings = Settings(api_key="S3cret@123!x")

        with caplog.at_level(logging.DEBUG):
            result = settings._load_with_masking("api_key")

        assert result == "S3cret@123!x"
        assert len(caplog.records) == 1
        assert "api_key" in caplog.records[0].message
        # S3cret@123!x is 12 chars, first 4 shown = 8 asterisks
        assert "S3cr********" in caplog.records[0].message

    def test_log_safe_config(self):
        """Test safe configuration dict with masked secrets."""
        settings = Settings(
            session_id="default",
            api_key="MyS3cur3P@ss!x",
            admin_api_key="sk-12345678",
        )

        safe_config = settings.log_safe_config()

        # Check that secrets are masked
        assert "MyS3cur3P@ss!x" not in str(safe_config)
        assert "sk-12345678" not in str(safe_config)
        # MyS3cur3P@ss!x is 14 chars, first 4 shown = 10 asterisks
        assert safe_config["api_key"] == "MyS3**********"
        # sk-12345678 is 11 chars, first 4 shown = 7 asterisks
        assert safe_config["admin_api_key"] == "sk-1*******"

        # Check that non-secrets are present
        assert safe_config["session_id"] == "default"

    def test_log_safe_config_no_api_key(self):
        """Test safe logging when API key not set."""
        settings = Settings(
            api_key=None,
        )

        safe_config = settings.log_safe_config()

        # API key should be None in output
        assert safe_config["api_key"] is None

    def test_log_config_info(self, caplog):
        """Test log_config_info method for logging."""
        settings = Settings(
            session_id="default",
        )

        with caplog.at_level(logging.INFO):
            settings.log_config_info()

        log_output = "\n".join([r.message for r in caplog.records])

        # Check that non-secrets are present
        assert "default" in log_output  # session_id

        # Check weights are logged
        assert "0.4" in log_output  # semantic weight
        assert "0.25" in log_output  # recency weight
        assert "0.2" in log_output  # outcome weight
        assert "0.15" in log_output  # importance weight

    def test_validate_permissions_enforce_production(self, tmp_path):
        """P3-SEC-L1: Test that validate_permissions enforces in production."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value")
        env_file.chmod(0o644)

        with patch.dict(os.environ, {"T4DM_ENVIRONMENT": "production"}):
            with pytest.raises(PermissionError, match="permissive permissions"):
                Settings.validate_permissions(env_file)

    def test_validate_permissions_auto_fix(self, tmp_path):
        """P3-SEC-L1: Test that validate_permissions can auto-fix."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value")
        env_file.chmod(0o644)

        Settings.validate_permissions(env_file, auto_fix=True)

        # Permissions should be fixed
        assert (env_file.stat().st_mode & 0o777) == 0o600


class TestEnvironmentOverrides:
    """Test environment variable overrides with masking."""

    def test_api_key_override(self):
        """Test API key override from environment."""
        with patch.dict(os.environ, {"T4DM_API_KEY": "env_api_key"}):
            settings = Settings()
            assert settings.api_key == "env_api_key"

    def test_admin_api_key_override(self):
        """Test admin API key override from environment."""
        with patch.dict(os.environ, {"T4DM_ADMIN_API_KEY": "env_admin_key"}):
            settings = Settings()
            assert settings.admin_api_key == "env_admin_key"

    def test_weights_override_validation(self):
        """Test that weight overrides are validated."""
        with patch.dict(
            os.environ,
            {
                "T4DM_NEO4J_PASSWORD": "ValidP@ss123!x",  # 14 chars, 4 classes
                "T4DM_EPISODIC_WEIGHT_SEMANTIC": "0.5",
                "T4DM_EPISODIC_WEIGHT_RECENCY": "0.3",
                "T4DM_EPISODIC_WEIGHT_OUTCOME": "0.1",
                "T4DM_EPISODIC_WEIGHT_IMPORTANCE": "0.05",  # Sum = 0.95
            },
        ):
            with pytest.raises(ValueError, match="must sum to 1.0"):
                Settings()
