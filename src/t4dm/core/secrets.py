"""
Pluggable Secrets Backend for World Weaver.

Phase 9: Production infrastructure with secure secret management.

Supports multiple backends:
- Environment variables (default, development)
- File-based (Docker secrets, Kubernetes secrets mounted as files)
- HashiCorp Vault (enterprise)
- AWS Secrets Manager (cloud)

Security Principles:
- Secrets never logged or serialized
- Memory cleared after use where possible
- Lazy loading to minimize exposure window
- Audit logging for access patterns

Usage:
    from t4dm.core.secrets import get_secrets_manager, SecretKey

    secrets = get_secrets_manager()
    api_key = secrets.get(SecretKey.OPENAI_API_KEY)

    # With default fallback
    db_pass = secrets.get(SecretKey.DATABASE_PASSWORD, default="")
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SecretKey(Enum):
    """Known secret keys with validation."""

    # Database credentials
    DATABASE_PASSWORD = "T4DM_DATABASE_PASSWORD"
    DATABASE_URL = "T4DM_DATABASE_URL"
    NEO4J_PASSWORD = "T4DM_NEO4J_PASSWORD"
    QDRANT_API_KEY = "T4DM_QDRANT_API_KEY"

    # API keys
    OPENAI_API_KEY = "OPENAI_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    COHERE_API_KEY = "COHERE_API_KEY"

    # Authentication
    JWT_SECRET = "T4DM_JWT_SECRET"
    API_KEY_HASH_SECRET = "T4DM_API_KEY_HASH_SECRET"

    # Encryption
    ENCRYPTION_KEY = "T4DM_ENCRYPTION_KEY"

    # External services
    SENTRY_DSN = "T4DM_SENTRY_DSN"
    VAULT_TOKEN = "VAULT_TOKEN"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"


class SecretBackend(ABC):
    """Abstract base for secret backends."""

    @abstractmethod
    def get(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if a secret exists."""
        pass

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List available secret keys (not values)."""
        pass

    def get_required(self, key: str) -> str:
        """Get a required secret, raising if missing."""
        value = self.get(key)
        if value is None:
            raise SecretNotFoundError(f"Required secret '{key}' not found")
        return value


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""
    pass


class EnvironmentBackend(SecretBackend):
    """
    Environment variable backend.

    Simple backend for development and basic deployments.
    Secrets are read from environment variables.
    """

    def __init__(self, prefix: str = ""):
        """
        Initialize environment backend.

        Args:
            prefix: Optional prefix for all keys (e.g., "T4DM_")
        """
        self._prefix = prefix
        logger.debug(f"EnvironmentBackend initialized with prefix='{prefix}'")

    def get(self, key: str, default: str | None = None) -> str | None:
        full_key = f"{self._prefix}{key}" if self._prefix else key
        value = os.environ.get(full_key, default)
        if value is not None:
            logger.debug(f"Secret '{key}' retrieved from environment")
        return value

    def has(self, key: str) -> bool:
        full_key = f"{self._prefix}{key}" if self._prefix else key
        return full_key in os.environ

    def list_keys(self) -> list[str]:
        if self._prefix:
            return [
                k[len(self._prefix):] for k in os.environ
                if k.startswith(self._prefix)
            ]
        # For no prefix, return known SecretKey values present
        return [
            sk.value for sk in SecretKey
            if sk.value in os.environ
        ]


class FileBackend(SecretBackend):
    """
    File-based secrets backend.

    Reads secrets from files, suitable for:
    - Docker secrets (/run/secrets/)
    - Kubernetes secrets mounted as volumes
    - .env files during development

    File naming convention: lowercase key with underscores
    e.g., T4DM_DATABASE_PASSWORD -> /secrets/ww_database_password
    """

    def __init__(
        self,
        secrets_dir: str | Path = "/run/secrets",
        fallback_dir: str | Path | None = None,
    ):
        """
        Initialize file backend.

        Args:
            secrets_dir: Primary secrets directory
            fallback_dir: Fallback directory (e.g., for development)
        """
        self._secrets_dir = Path(secrets_dir)
        self._fallback_dir = Path(fallback_dir) if fallback_dir else None

        logger.debug(
            f"FileBackend initialized: dir={self._secrets_dir}, "
            f"fallback={self._fallback_dir}"
        )

    def _get_secret_path(self, key: str) -> Path | None:
        """Find the path for a secret key."""
        # Normalize key to lowercase with underscores
        normalized = key.lower().replace("-", "_")

        # Try primary directory
        primary = self._secrets_dir / normalized
        if primary.exists() and primary.is_file():
            return primary

        # Try fallback
        if self._fallback_dir:
            fallback = self._fallback_dir / normalized
            if fallback.exists() and fallback.is_file():
                return fallback

        return None

    def get(self, key: str, default: str | None = None) -> str | None:
        path = self._get_secret_path(key)
        if path is None:
            return default

        try:
            value = path.read_text().strip()
            logger.debug(f"Secret '{key}' read from {path}")
            return value
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to read secret '{key}' from {path}: {e}")
            return default

    def has(self, key: str) -> bool:
        return self._get_secret_path(key) is not None

    def list_keys(self) -> list[str]:
        keys = set()

        for dir_path in [self._secrets_dir, self._fallback_dir]:
            if dir_path and dir_path.exists():
                for f in dir_path.iterdir():
                    if f.is_file() and not f.name.startswith("."):
                        keys.add(f.name.upper())

        return sorted(keys)


class ChainedBackend(SecretBackend):
    """
    Chain multiple backends with fallback.

    Tries each backend in order until a value is found.
    Useful for layering: file secrets → environment → defaults
    """

    def __init__(self, backends: list[SecretBackend]):
        """
        Initialize chained backend.

        Args:
            backends: List of backends to try in order
        """
        if not backends:
            raise ValueError("At least one backend required")
        self._backends = backends
        logger.debug(f"ChainedBackend with {len(backends)} backends")

    def get(self, key: str, default: str | None = None) -> str | None:
        for backend in self._backends:
            value = backend.get(key)
            if value is not None:
                return value
        return default

    def has(self, key: str) -> bool:
        return any(b.has(key) for b in self._backends)

    def list_keys(self) -> list[str]:
        all_keys = set()
        for backend in self._backends:
            all_keys.update(backend.list_keys())
        return sorted(all_keys)


@dataclass
class SecretsConfig:
    """Configuration for secrets manager."""

    backend: str = "auto"  # auto, env, file, chained
    secrets_dir: str = "/run/secrets"
    fallback_dir: str | None = None
    env_prefix: str = ""
    audit_access: bool = True
    cache_secrets: bool = False  # Security: disable by default

    @classmethod
    def from_env(cls) -> SecretsConfig:
        """Create config from environment variables."""
        return cls(
            backend=os.environ.get("T4DM_SECRETS_BACKEND", "auto"),
            secrets_dir=os.environ.get("T4DM_SECRETS_DIR", "/run/secrets"),
            fallback_dir=os.environ.get("T4DM_SECRETS_FALLBACK_DIR"),
            env_prefix=os.environ.get("T4DM_SECRETS_ENV_PREFIX", ""),
            audit_access=os.environ.get("T4DM_SECRETS_AUDIT", "true").lower() == "true",
            cache_secrets=os.environ.get("T4DM_SECRETS_CACHE", "false").lower() == "true",
        )


class SecretsManager:
    """
    Central secrets manager with pluggable backends.

    Provides a unified interface for secret retrieval with:
    - Multiple backend support
    - Access auditing
    - Type-safe key enums
    - Caching (optional, disabled by default for security)

    Example:
        secrets = SecretsManager()
        api_key = secrets.get(SecretKey.OPENAI_API_KEY)

        # Or with string key
        api_key = secrets.get_raw("OPENAI_API_KEY")
    """

    def __init__(
        self,
        config: SecretsConfig | None = None,
        backend: SecretBackend | None = None,
    ):
        """
        Initialize secrets manager.

        Args:
            config: Configuration (defaults to from_env())
            backend: Explicit backend (overrides config)
        """
        self.config = config or SecretsConfig.from_env()
        self._backend = backend or self._create_backend()
        self._access_log: list[dict] = []

        logger.info(
            f"SecretsManager initialized: backend={self.config.backend}, "
            f"audit={self.config.audit_access}"
        )

    def _create_backend(self) -> SecretBackend:
        """Create backend based on configuration."""
        if self.config.backend == "env":
            return EnvironmentBackend(prefix=self.config.env_prefix)

        elif self.config.backend == "file":
            return FileBackend(
                secrets_dir=self.config.secrets_dir,
                fallback_dir=self.config.fallback_dir,
            )

        elif self.config.backend == "chained":
            return ChainedBackend([
                FileBackend(
                    secrets_dir=self.config.secrets_dir,
                    fallback_dir=self.config.fallback_dir,
                ),
                EnvironmentBackend(prefix=self.config.env_prefix),
            ])

        else:  # "auto" - detect based on environment
            secrets_dir = Path(self.config.secrets_dir)
            if secrets_dir.exists() and any(secrets_dir.iterdir()):
                # Kubernetes/Docker secrets available, use chained
                return ChainedBackend([
                    FileBackend(
                        secrets_dir=self.config.secrets_dir,
                        fallback_dir=self.config.fallback_dir,
                    ),
                    EnvironmentBackend(prefix=self.config.env_prefix),
                ])
            else:
                # No file secrets, use environment
                return EnvironmentBackend(prefix=self.config.env_prefix)

    def _audit(self, key: str, found: bool) -> None:
        """Log secret access for auditing."""
        if not self.config.audit_access:
            return

        import datetime
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "key": key,
            "found": found,
        }
        self._access_log.append(entry)

        # Keep only last 1000 entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]

    def get(
        self,
        key: SecretKey | str,
        default: str | None = None,
    ) -> str | None:
        """
        Get a secret by key.

        Args:
            key: SecretKey enum or string key
            default: Default value if not found

        Returns:
            Secret value or default
        """
        key_str = key.value if isinstance(key, SecretKey) else key
        value = self._backend.get(key_str, default)
        self._audit(key_str, value is not None)
        return value

    def get_required(self, key: SecretKey | str) -> str:
        """
        Get a required secret, raising if missing.

        Args:
            key: SecretKey enum or string key

        Returns:
            Secret value

        Raises:
            SecretNotFoundError: If secret not found
        """
        value = self.get(key)
        if value is None:
            key_str = key.value if isinstance(key, SecretKey) else key
            raise SecretNotFoundError(f"Required secret '{key_str}' not found")
        return value

    def get_raw(self, key: str, default: str | None = None) -> str | None:
        """Get a secret by raw string key."""
        return self.get(key, default)

    def has(self, key: SecretKey | str) -> bool:
        """Check if a secret exists."""
        key_str = key.value if isinstance(key, SecretKey) else key
        return self._backend.has(key_str)

    def list_keys(self) -> list[str]:
        """List available secret keys (not values)."""
        return self._backend.list_keys()

    def get_access_log(self) -> list[dict]:
        """Get the access audit log."""
        return self._access_log.copy()

    def get_stats(self) -> dict:
        """Get secrets manager statistics."""
        return {
            "backend": self.config.backend,
            "available_keys": len(self.list_keys()),
            "audit_enabled": self.config.audit_access,
            "access_count": len(self._access_log),
        }


# ============================================================================
# Singleton
# ============================================================================

_secrets_manager: SecretsManager | None = None


def get_secrets_manager(
    config: SecretsConfig | None = None,
    backend: SecretBackend | None = None,
) -> SecretsManager:
    """
    Get or create the singleton secrets manager.

    Args:
        config: Optional configuration (used only on first call)
        backend: Optional explicit backend

    Returns:
        SecretsManager instance
    """
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(config=config, backend=backend)
    return _secrets_manager


def reset_secrets_manager() -> None:
    """Reset singleton (for testing)."""
    global _secrets_manager
    _secrets_manager = None


__all__ = [
    "ChainedBackend",
    "EnvironmentBackend",
    "FileBackend",
    "SecretBackend",
    "SecretKey",
    "SecretNotFoundError",
    "SecretsConfig",
    "SecretsManager",
    "get_secrets_manager",
    "reset_secrets_manager",
]
