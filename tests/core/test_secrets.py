"""
Tests for Secrets Backend (Phase 9).

Tests the pluggable secrets management system.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ww.core.secrets import (
    ChainedBackend,
    EnvironmentBackend,
    FileBackend,
    SecretKey,
    SecretNotFoundError,
    SecretsConfig,
    SecretsManager,
    get_secrets_manager,
    reset_secrets_manager,
)


class TestEnvironmentBackend:
    """Tests for EnvironmentBackend."""

    def test_get_existing(self):
        """Test getting an existing environment variable."""
        with patch.dict(os.environ, {"TEST_SECRET": "secret_value"}):
            backend = EnvironmentBackend()
            assert backend.get("TEST_SECRET") == "secret_value"

    def test_get_missing(self):
        """Test getting a missing variable returns default."""
        backend = EnvironmentBackend()
        assert backend.get("NONEXISTENT_SECRET_12345") is None
        assert backend.get("NONEXISTENT_SECRET_12345", default="default") == "default"

    def test_get_with_prefix(self):
        """Test getting with a prefix."""
        with patch.dict(os.environ, {"WW_TEST_SECRET": "prefixed_value"}):
            backend = EnvironmentBackend(prefix="WW_")
            assert backend.get("TEST_SECRET") == "prefixed_value"

    def test_has_existing(self):
        """Test has returns True for existing."""
        with patch.dict(os.environ, {"TEST_SECRET": "value"}):
            backend = EnvironmentBackend()
            assert backend.has("TEST_SECRET") is True

    def test_has_missing(self):
        """Test has returns False for missing."""
        backend = EnvironmentBackend()
        assert backend.has("NONEXISTENT_SECRET_12345") is False

    def test_list_keys_with_prefix(self):
        """Test listing keys with prefix."""
        with patch.dict(os.environ, {
            "WW_SECRET1": "v1",
            "WW_SECRET2": "v2",
            "OTHER_VAR": "v3",
        }, clear=True):
            backend = EnvironmentBackend(prefix="WW_")
            keys = backend.list_keys()
            assert "SECRET1" in keys
            assert "SECRET2" in keys
            assert "OTHER_VAR" not in keys


class TestFileBackend:
    """Tests for FileBackend."""

    def test_get_existing_file(self):
        """Test getting an existing secret file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "test_secret"
            secret_path.write_text("file_secret_value\n")

            backend = FileBackend(secrets_dir=tmpdir)
            assert backend.get("TEST_SECRET") == "file_secret_value"

    def test_get_missing_file(self):
        """Test getting a missing file returns default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend(secrets_dir=tmpdir)
            assert backend.get("NONEXISTENT") is None
            assert backend.get("NONEXISTENT", default="def") == "def"

    def test_get_with_fallback(self):
        """Test fallback directory."""
        with tempfile.TemporaryDirectory() as primary:
            with tempfile.TemporaryDirectory() as fallback:
                # Secret in fallback only
                secret_path = Path(fallback) / "fallback_secret"
                secret_path.write_text("fallback_value")

                backend = FileBackend(secrets_dir=primary, fallback_dir=fallback)
                assert backend.get("FALLBACK_SECRET") == "fallback_value"

    def test_has_existing(self):
        """Test has returns True for existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "exists"
            secret_path.write_text("value")

            backend = FileBackend(secrets_dir=tmpdir)
            assert backend.has("EXISTS") is True

    def test_has_missing(self):
        """Test has returns False for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend(secrets_dir=tmpdir)
            assert backend.has("NONEXISTENT") is False

    def test_list_keys(self):
        """Test listing available keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "secret1").write_text("v1")
            (Path(tmpdir) / "secret2").write_text("v2")

            backend = FileBackend(secrets_dir=tmpdir)
            keys = backend.list_keys()
            assert "SECRET1" in keys
            assert "SECRET2" in keys


class TestChainedBackend:
    """Tests for ChainedBackend."""

    def test_chained_priority(self):
        """Test first backend takes priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File backend has the value
            secret_path = Path(tmpdir) / "priority_secret"
            secret_path.write_text("file_value")

            file_backend = FileBackend(secrets_dir=tmpdir)
            env_backend = EnvironmentBackend()

            with patch.dict(os.environ, {"PRIORITY_SECRET": "env_value"}):
                chained = ChainedBackend([file_backend, env_backend])
                # File backend is first, should return file value
                assert chained.get("PRIORITY_SECRET") == "file_value"

    def test_chained_fallback(self):
        """Test fallback to second backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_backend = FileBackend(secrets_dir=tmpdir)  # Empty
            env_backend = EnvironmentBackend()

            with patch.dict(os.environ, {"FALLBACK_SECRET": "env_value"}):
                chained = ChainedBackend([file_backend, env_backend])
                assert chained.get("FALLBACK_SECRET") == "env_value"

    def test_chained_has(self):
        """Test has checks all backends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_backend = FileBackend(secrets_dir=tmpdir)
            env_backend = EnvironmentBackend()

            with patch.dict(os.environ, {"ENV_ONLY": "value"}):
                chained = ChainedBackend([file_backend, env_backend])
                assert chained.has("ENV_ONLY") is True
                assert chained.has("NONEXISTENT") is False

    def test_empty_backends_raises(self):
        """Test empty backends list raises."""
        with pytest.raises(ValueError):
            ChainedBackend([])


class TestSecretsConfig:
    """Tests for SecretsConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SecretsConfig()
        assert config.backend == "auto"
        assert config.secrets_dir == "/run/secrets"
        assert config.audit_access is True
        assert config.cache_secrets is False

    def test_from_env(self):
        """Test configuration from environment."""
        with patch.dict(os.environ, {
            "WW_SECRETS_BACKEND": "file",
            "WW_SECRETS_DIR": "/custom/secrets",
            "WW_SECRETS_AUDIT": "false",
        }):
            config = SecretsConfig.from_env()
            assert config.backend == "file"
            assert config.secrets_dir == "/custom/secrets"
            assert config.audit_access is False


class TestSecretsManager:
    """Tests for SecretsManager."""

    @pytest.fixture
    def manager(self):
        """Create fresh secrets manager."""
        reset_secrets_manager()
        return SecretsManager(
            config=SecretsConfig(backend="env"),
        )

    def test_get_by_enum(self, manager):
        """Test getting by SecretKey enum."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            value = manager.get(SecretKey.OPENAI_API_KEY)
            assert value == "sk-test"

    def test_get_by_string(self, manager):
        """Test getting by string key."""
        with patch.dict(os.environ, {"CUSTOM_SECRET": "custom_value"}):
            value = manager.get("CUSTOM_SECRET")
            assert value == "custom_value"

    def test_get_required_exists(self, manager):
        """Test get_required with existing secret."""
        with patch.dict(os.environ, {"REQUIRED_SECRET": "exists"}):
            value = manager.get_required("REQUIRED_SECRET")
            assert value == "exists"

    def test_get_required_missing(self, manager):
        """Test get_required raises for missing."""
        with pytest.raises(SecretNotFoundError):
            manager.get_required("DEFINITELY_MISSING_12345")

    def test_has(self, manager):
        """Test has method."""
        with patch.dict(os.environ, {"EXISTS": "yes"}):
            assert manager.has("EXISTS") is True
            assert manager.has("MISSING") is False

    def test_list_keys(self, manager):
        """Test listing keys."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            keys = manager.list_keys()
            assert isinstance(keys, list)

    def test_audit_log(self, manager):
        """Test access audit logging."""
        with patch.dict(os.environ, {"AUDIT_TEST": "value"}):
            manager.get("AUDIT_TEST")
            manager.get("MISSING_AUDIT")

            log = manager.get_access_log()
            assert len(log) == 2
            assert log[0]["key"] == "AUDIT_TEST"
            assert log[0]["found"] is True
            assert log[1]["key"] == "MISSING_AUDIT"
            assert log[1]["found"] is False

    def test_stats(self, manager):
        """Test statistics."""
        stats = manager.get_stats()
        assert "backend" in stats
        assert "available_keys" in stats
        assert "audit_enabled" in stats


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_secrets_manager_singleton(self):
        """Test singleton creation."""
        reset_secrets_manager()
        m1 = get_secrets_manager()
        m2 = get_secrets_manager()
        assert m1 is m2

    def test_reset_secrets_manager(self):
        """Test singleton reset."""
        m1 = get_secrets_manager()
        reset_secrets_manager()
        m2 = get_secrets_manager()
        assert m1 is not m2


class TestSecretKey:
    """Tests for SecretKey enum."""

    def test_known_keys(self):
        """Test known secret keys exist."""
        assert SecretKey.OPENAI_API_KEY.value == "OPENAI_API_KEY"
        assert SecretKey.DATABASE_PASSWORD.value == "WW_DATABASE_PASSWORD"
        assert SecretKey.JWT_SECRET.value == "WW_JWT_SECRET"

    def test_all_keys_have_values(self):
        """Test all keys have string values."""
        for key in SecretKey:
            assert isinstance(key.value, str)
            assert len(key.value) > 0
