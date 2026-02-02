"""Core types, configuration, and protocols for T4DM."""

from t4dm.core.cache import (
    CacheStats,
    InMemoryCache,
    RedisCache,
    close_cache,
    get_cache,
    hash_query,
    hash_text,
    reset_cache,
)
from t4dm.core.cache_config import CacheTier, CacheTierConfig, RedisCacheConfig
from t4dm.core.config import Settings, get_settings

# Phase 9: Production infrastructure
from t4dm.core.emergency import (
    CircuitBreaker,
    CircuitBreakerConfig,
    EmergencyManager,
    PanicLevel,
    TrackedRequest,
    get_emergency_manager,
    reset_emergency_manager,
)
from t4dm.core.feature_flags import (
    FeatureFlag,
    FeatureFlags,
    FlagConfig,
    disable_feature,
    enable_feature,
    get_feature_flags,
    is_feature_enabled,
    reset_feature_flags,
)
from t4dm.core.protocols import (
    EmbeddingProvider,
    GraphStore,
    VectorStore,
)
from t4dm.core.secrets import (
    SecretBackend,
    SecretKey,
    SecretNotFoundError,
    SecretsConfig,
    SecretsManager,
    get_secrets_manager,
    reset_secrets_manager,
)
from t4dm.core.services import (
    RateLimiter,
    cleanup_services,
    get_services,
    reset_services,
)
from t4dm.core.types import (
    ConsolidationEvent,
    Domain,
    Entity,
    EntityType,
    Episode,
    Outcome,
    Procedure,
    Relationship,
    RelationType,
)
from t4dm.core.validation import (
    SessionValidationError,
    ValidationError,
    sanitize_session_id,
    sanitize_string,
    validate_session_id,
)

__all__ = [
    # Cache (Phase 3A)
    "CacheStats",
    "CacheTier",
    "CacheTierConfig",
    "InMemoryCache",
    "RedisCache",
    "RedisCacheConfig",
    "close_cache",
    "get_cache",
    "hash_query",
    "hash_text",
    "reset_cache",
    # Config
    "Settings",
    "get_settings",
    # Emergency (Phase 9)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "EmergencyManager",
    "PanicLevel",
    "TrackedRequest",
    "get_emergency_manager",
    "reset_emergency_manager",
    # Feature Flags (Phase 9)
    "FeatureFlag",
    "FeatureFlags",
    "FlagConfig",
    "disable_feature",
    "enable_feature",
    "get_feature_flags",
    "is_feature_enabled",
    "reset_feature_flags",
    # Protocols
    "EmbeddingProvider",
    "GraphStore",
    "VectorStore",
    # Secrets (Phase 9)
    "SecretBackend",
    "SecretKey",
    "SecretNotFoundError",
    "SecretsConfig",
    "SecretsManager",
    "get_secrets_manager",
    "reset_secrets_manager",
    # Types
    "ConsolidationEvent",
    "Domain",
    "Entity",
    "EntityType",
    "Episode",
    "Outcome",
    "Procedure",
    "RelationType",
    "Relationship",
    # Services
    "RateLimiter",
    "cleanup_services",
    "get_services",
    "reset_services",
    # Validation
    "SessionValidationError",
    "ValidationError",
    "sanitize_session_id",
    "sanitize_string",
    "validate_session_id",
]
