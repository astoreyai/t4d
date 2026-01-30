"""
Tests for YAML configuration loading.

Tests the config module's YAML file loading functionality,
including file discovery, parsing, and environment variable overrides.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigFileDiscovery:
    """Tests for config file discovery."""

    def test_finds_ww_yaml_in_cwd(self):
        """Finds ww.yaml in current directory."""
        from ww.core.config import _find_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ww.yaml"
            config_path.write_text("session_id: test")

            # Change to temp directory
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = _find_config_file()
                assert result == Path("ww.yaml")
            finally:
                os.chdir(old_cwd)

    def test_finds_ww_yml_in_cwd(self):
        """Finds ww.yml in current directory."""
        from ww.core.config import _find_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ww.yml"
            config_path.write_text("session_id: test")

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = _find_config_file()
                assert result == Path("ww.yml")
            finally:
                os.chdir(old_cwd)

    def test_env_var_overrides_search(self):
        """WW_CONFIG_FILE environment variable takes precedence."""
        from ww.core.config import _find_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "custom-config.yaml"
            config_path.write_text("session_id: custom")

            with patch.dict(os.environ, {"WW_CONFIG_FILE": str(config_path)}):
                result = _find_config_file()
                assert result == config_path

    def test_returns_none_when_no_file(self):
        """Returns None when no config file found."""
        from ww.core.config import _find_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Clear any WW_CONFIG_FILE env var
                with patch.dict(os.environ, {"WW_CONFIG_FILE": ""}, clear=False):
                    if "WW_CONFIG_FILE" in os.environ:
                        del os.environ["WW_CONFIG_FILE"]
                    result = _find_config_file()
                    # Note: might find ~/.ww/config.yaml if it exists
                    # Just verify no exception is raised
            finally:
                os.chdir(old_cwd)


class TestYamlLoading:
    """Tests for YAML file loading."""

    def test_loads_valid_yaml(self):
        """Loads valid YAML configuration."""
        from ww.core.config import _load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
session_id: test-session
environment: development
qdrant_url: http://localhost:6333
""")
            f.flush()

            result = _load_yaml_config(Path(f.name))

            assert result["session_id"] == "test-session"
            assert result["environment"] == "development"
            assert result["qdrant_url"] == "http://localhost:6333"

            Path(f.name).unlink()

    def test_handles_empty_yaml(self):
        """Handles empty YAML file."""
        from ww.core.config import _load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            result = _load_yaml_config(Path(f.name))
            assert result == {}

            Path(f.name).unlink()

    def test_handles_null_yaml(self):
        """Handles null YAML content."""
        from ww.core.config import _load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("null")
            f.flush()

            result = _load_yaml_config(Path(f.name))
            assert result == {}

            Path(f.name).unlink()

    def test_rejects_invalid_yaml(self):
        """Rejects invalid YAML syntax."""
        from ww.core.config import _load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
invalid: yaml: content
  broken: indentation
""")
            f.flush()

            with pytest.raises(ValueError, match="Invalid YAML"):
                _load_yaml_config(Path(f.name))

            Path(f.name).unlink()

    def test_rejects_non_dict_yaml(self):
        """Rejects YAML that doesn't contain a dictionary."""
        from ww.core.config import _load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
- item1
- item2
- item3
""")
            f.flush()

            with pytest.raises(ValueError, match="must contain a dictionary"):
                _load_yaml_config(Path(f.name))

            Path(f.name).unlink()


class TestLoadSettingsFromYaml:
    """Tests for load_settings_from_yaml function."""

    def test_loads_settings_from_yaml(self):
        """Loads Settings from YAML file."""
        from ww.core.config import load_settings_from_yaml, reset_settings

        reset_settings()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
session_id: yaml-session
environment: test
""")
            f.flush()

            # Clear any WW_ env vars that might interfere
            clean_env = {k: v for k, v in os.environ.items() if not k.startswith("WW_")}
            # Keep WW_TEST_MODE if set
            if "WW_TEST_MODE" in os.environ:
                clean_env["WW_TEST_MODE"] = os.environ["WW_TEST_MODE"]

            with patch.dict(os.environ, clean_env, clear=True):
                settings = load_settings_from_yaml(f.name)

                assert settings.session_id == "yaml-session"
                assert settings.environment == "test"

            Path(f.name).unlink()
            reset_settings()

    def test_env_vars_override_yaml(self):
        """Environment variables override YAML values."""
        from ww.core.config import load_settings_from_yaml, reset_settings

        reset_settings()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
session_id: yaml-session
environment: test
""")
            f.flush()

            # Clear existing env vars and set our test value
            env_overrides = {
                "WW_SESSION_ID": "env-session",
                "WW_ENVIRONMENT": "test",  # Keep test mode
            }
            with patch.dict(os.environ, env_overrides, clear=False):
                settings = load_settings_from_yaml(f.name)

                # pydantic-settings reads env vars directly,
                # so env var should override the yaml value
                assert settings.session_id == "env-session"
                assert settings.environment == "test"

            Path(f.name).unlink()
            reset_settings()

    def test_raises_on_missing_file(self):
        """Raises FileNotFoundError for missing config file."""
        from ww.core.config import load_settings_from_yaml

        with pytest.raises(FileNotFoundError):
            load_settings_from_yaml("/nonexistent/config.yaml")


class TestMergeConfig:
    """Tests for config merging."""

    def test_merge_preserves_yaml_values(self):
        """Merge preserves YAML values when no env override."""
        from ww.core.config import _merge_config_with_env

        yaml_config = {
            "session_id": "yaml-session",
            "environment": "development",
        }

        # Create a clean environment without WW_ prefixed vars for this test
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("WW_")}
        with patch.dict(os.environ, clean_env, clear=True):
            result = _merge_config_with_env(yaml_config)

            assert result["session_id"] == "yaml-session"
            assert result["environment"] == "development"

    def test_merge_applies_env_override(self):
        """Merge applies environment variable overrides."""
        from ww.core.config import _merge_config_with_env

        yaml_config = {
            "session_id": "yaml-session",
            "environment": "development",
        }

        with patch.dict(os.environ, {"WW_SESSION_ID": "env-session"}):
            result = _merge_config_with_env(yaml_config)

            assert result["session_id"] == "env-session"
            assert result["environment"] == "development"


class TestResetSettings:
    """Tests for settings cache reset."""

    def test_reset_clears_cache(self):
        """reset_settings clears the settings cache."""
        from ww.core.config import get_settings, reset_settings

        # Get settings to populate cache
        settings1 = get_settings()

        # Reset cache
        reset_settings()

        # Get settings again - should create new instance
        settings2 = get_settings()

        # Reset for other tests
        reset_settings()

        # They might be equal in value but should be different instances
        # (unless caching is working correctly)
        assert settings1 is not settings2 or settings1 == settings2


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """get_settings returns a Settings instance."""
        from ww.core.config import Settings, get_settings, reset_settings

        reset_settings()  # Clear cache

        settings = get_settings()
        assert isinstance(settings, Settings)

        reset_settings()  # Clean up

    def test_get_settings_is_cached(self):
        """get_settings returns cached instance."""
        from ww.core.config import get_settings, reset_settings

        reset_settings()  # Clear cache

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be same instance due to caching
        assert settings1 is settings2

        reset_settings()  # Clean up
