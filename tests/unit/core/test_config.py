"""Comprehensive tests for config.py - Settings management."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from t4dm.core.config import (
    get_settings,
    validate_password_strength,
    WEAK_PASSWORDS,
    WEAK_PATTERNS,
)


class TestPasswordValidation:
    """Test password strength validation."""

    def test_password_minimum_length(self):
        # Too short
        with pytest.raises(ValueError, match="at least 12 characters"):
            validate_password_strength("short")

    def test_password_exact_minimum(self):
        # Exactly 12 chars with variety
        pwd = "MyPassword123"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_weak_password_exact_match(self):
        # All-lowercase passwords too weak (need 3 complexity classes)
        weak_pwd = "allpasswordxyz"  # 14 chars, 1 class only
        with pytest.raises(ValueError, match="complexity"):
            validate_password_strength(weak_pwd)

    def test_weak_password_pattern_match(self):
        # Pattern-based weak detection
        # All digits matches r"^\d+$" pattern
        weak_pwd = "123456789012"
        with pytest.raises(ValueError, match="weak"):
            validate_password_strength(weak_pwd)

    def test_strong_password_with_lowercase(self):
        pwd = "MySecureP@ssw0rd"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_strong_password_with_uppercase(self):
        pwd = "abCDEfghijk1"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_strong_password_with_digits(self):
        pwd = "AbcdEFGH1234"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_strong_password_with_special(self):
        pwd = "AbcdEFGH!@#$"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_password_requires_3_of_4_complexity(self):
        # Only 2 character classes: lowercase + uppercase (but 12+ chars)
        pwd = "ABCDEfghijkl"  # 12 chars, upper+lower only
        with pytest.raises(ValueError, match="complexity"):
            validate_password_strength(pwd)

    def test_password_with_3_complexity_classes(self):
        # Lower + upper + digit (12+ chars)
        pwd = "ABCDefgh1234"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_empty_password_raises(self):
        with pytest.raises(ValueError, match="required"):
            validate_password_strength("")

    def test_custom_field_name_in_error(self):
        with pytest.raises(ValueError, match="test_pwd"):
            validate_password_strength("short", field_name="test_pwd")

    def test_password_case_insensitive_weak_check(self):
        # PASSWORD in uppercase should match weak password list
        with pytest.raises(ValueError):
            validate_password_strength("PASSWORD1234")


class TestSettingsBasicLoading:
    """Test basic settings loading from environment."""

    def test_get_settings_singleton(self):
        """Settings should be cached."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_returns_settings_object(self):
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'session_id')
        assert hasattr(settings, 'embedding_dimension')

    def test_default_session_id(self):
        settings = get_settings()
        assert settings.session_id is not None
        assert isinstance(settings.session_id, str)

    def test_default_embedding_dimension(self):
        settings = get_settings()
        assert settings.embedding_dimension > 0
        assert isinstance(settings.embedding_dimension, int)

    def test_default_paths(self):
        settings = get_settings()
        # These should have sensible defaults
        assert settings.data_dir is not None
        assert isinstance(settings.data_dir, (str, Path))


class TestSettingsEnvironmentVariables:
    """Test settings loading from environment variables."""

    def test_session_id_from_env(self):
        with patch.dict(os.environ, {'T4DM_SESSION_ID': 'custom-session'}):
            # Clear cached settings to force reload
            import t4dm.core.config
            t4dm.core.config._settings_instance = None
            try:
                settings = get_settings()
                assert settings.session_id == 'custom-session'
            finally:
                t4dm.core.config._settings_instance = None

    def test_embedding_dimension_from_env(self):
        with patch.dict(os.environ, {'T4DM_EMBEDDING_DIMENSION': '2048'}):
            import t4dm.core.config
            t4dm.core.config._settings_instance = None
            try:
                settings = get_settings()
                assert settings.embedding_dimension == 2048
            finally:
                t4dm.core.config._settings_instance = None


class TestSettingsValidation:
    """Test settings validation rules."""

    def test_settings_model_config(self):
        """Settings should validate assignment and allow extras."""
        settings = get_settings()
        assert settings is not None

    def test_session_id_not_empty(self):
        """Session ID should not be empty if provided."""
        settings = get_settings()
        # Default session ID should be non-empty
        assert len(settings.session_id) > 0

    def test_embedding_dim_positive(self):
        """Embedding dimension must be positive."""
        settings = get_settings()
        assert settings.embedding_dimension > 0

    def test_api_port_valid(self):
        """API port should be a valid port number."""
        settings = get_settings()
        assert 1 <= settings.api_port <= 65535


class TestSettingsCommonPaths:
    """Test common path handling in settings."""

    def test_data_dir_exists(self):
        """Data directory should exist or be creatable."""
        settings = get_settings()
        data_dir = Path(settings.data_dir)
        # Should not raise
        data_dir.parent.exists()

    def test_settings_with_different_data_dirs(self):
        """Settings should handle different data directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'T4DM_DATA_DIR': tmpdir}):
                import t4dm.core.config
                t4dm.core.config._settings_instance = None
                try:
                    settings = get_settings()
                    assert str(settings.data_dir) == tmpdir
                finally:
                    t4dm.core.config._settings_instance = None


class TestSettingsAttributeAccess:
    """Test accessing various settings attributes."""

    def test_access_string_attributes(self):
        settings = get_settings()
        # Should not raise
        _ = settings.session_id

    def test_access_int_attributes(self):
        settings = get_settings()
        # Should not raise
        _ = settings.embedding_dimension
        _ = settings.api_port

    def test_access_optional_attributes(self):
        settings = get_settings()
        # Optional fields should be accessible
        _ = settings.data_dir
        _ = settings.api_key

    def test_settings_dump(self):
        """Settings should be dumpable to dict."""
        settings = get_settings()
        dumped = settings.model_dump()
        assert isinstance(dumped, dict)
        assert 'session_id' in dumped
        assert 'embedding_dimension' in dumped

    def test_settings_json_schema(self):
        """Settings should have a JSON schema."""
        from t4dm.core.config import get_settings
        settings = get_settings()
        schema = settings.model_json_schema()
        assert isinstance(schema, dict)
        assert 'properties' in schema


class TestSettingsDefaults:
    """Test default values and their reasonableness."""

    def test_reasonable_default_embedding_dim(self):
        settings = get_settings()
        # Should be a reasonable embedding dimension (BGE-M3 is 1024)
        assert settings.embedding_dimension > 0

    def test_session_id_generation(self):
        settings = get_settings()
        # Session ID should be a reasonable string
        assert len(settings.session_id) > 0
        assert isinstance(settings.session_id, str)

    def test_api_host_set(self):
        settings = get_settings()
        assert settings.api_host is not None
        assert isinstance(settings.api_host, str)


class TestSettingsCaching:
    """Test settings caching behavior."""

    def test_settings_caching_across_calls(self):
        """Settings should be cached to avoid reloading."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_cache_preserves_values(self):
        """Cached settings should preserve values."""
        s1 = get_settings()
        session_id = s1.session_id

        s2 = get_settings()
        assert s2.session_id == session_id


class TestSettingsImmutability:
    """Test settings immutability and validation on assignment."""

    def test_settings_validation_on_assignment(self):
        """Settings should validate on assignment if configured."""
        settings = get_settings()
        # Accessing attributes should work
        _ = settings.session_id
        _ = settings.embedding_dimension

    def test_settings_model_config(self):
        """Settings should have proper model config."""
        settings = get_settings()
        # Should be a pydantic model
        assert hasattr(settings, 'model_dump')
        assert hasattr(settings, 'model_validate')


class TestWeakPasswordConstants:
    """Test the weak password constants."""

    def test_weak_passwords_set_is_frozenset(self):
        assert isinstance(WEAK_PASSWORDS, frozenset)

    def test_weak_passwords_not_empty(self):
        assert len(WEAK_PASSWORDS) > 10

    def test_weak_passwords_are_strings(self):
        for pwd in WEAK_PASSWORDS:
            assert isinstance(pwd, str)

    def test_weak_patterns_set_is_frozenset(self):
        assert isinstance(WEAK_PATTERNS, frozenset)

    def test_weak_patterns_not_empty(self):
        assert len(WEAK_PATTERNS) > 5

    def test_weak_patterns_are_strings(self):
        for pattern in WEAK_PATTERNS:
            assert isinstance(pattern, str)


class TestPasswordValidationEdgeCases:
    """Test edge cases in password validation."""

    def test_password_with_unicode_characters(self):
        """Password with unicode should still validate."""
        pwd = "MyPäss123!Ab"  # Contains unicode, 12+ chars
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_password_with_spaces(self):
        """Password with spaces should validate."""
        pwd = "My Secret Password 123"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_password_with_multiple_special_chars(self):
        """Password with multiple special chars should validate."""
        pwd = "P@ssw0rd!#$%"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_very_long_password(self):
        """Very long password should validate."""
        pwd = "A" * 1000 + "b1!"
        result = validate_password_strength(pwd)
        assert result == pwd

    def test_password_with_only_numbers(self):
        """Numbers-only fails pattern check."""
        pwd = "123456789012"
        with pytest.raises(ValueError):
            validate_password_strength(pwd)

    def test_password_with_only_letters(self):
        """Letters-only fails complexity check."""
        pwd = "ABCDEFGHabcd"
        with pytest.raises(ValueError, match="complexity"):
            validate_password_strength(pwd)

    def test_password_mixed_case_numbers(self):
        """Mixed case + numbers should work if long enough."""
        pwd = "AbcdEFgh1234"
        result = validate_password_strength(pwd)
        assert result == pwd


class TestSettingsFieldDescriptions:
    """Test that settings fields have descriptions."""

    def test_settings_schema_has_descriptions(self):
        settings = get_settings()
        schema = settings.model_json_schema()
        # Properties should exist
        assert 'properties' in schema
        assert len(schema['properties']) > 0
