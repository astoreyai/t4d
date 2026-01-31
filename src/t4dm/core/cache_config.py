"""
Cache Configuration for World Weaver.

Defines cache settings for different tiers and environments.
"""

from dataclasses import dataclass, field
from enum import Enum


class CacheTier(Enum):
    """Cache tier types."""

    EMBEDDING = "embedding"
    SEARCH = "search"
    GRAPH = "graph"


@dataclass
class CacheTierConfig:
    """Configuration for a cache tier."""

    ttl: int  # Time-to-live in seconds
    max_size: int  # Max entries in fallback cache
    enabled: bool = True  # Whether this tier is enabled

    @classmethod
    def default_embedding(cls) -> "CacheTierConfig":
        """Default config for embedding cache."""
        return cls(
            ttl=3600,  # 1 hour - embeddings expensive to compute
            max_size=5000,  # ~5K embeddings in memory
            enabled=True,
        )

    @classmethod
    def default_search(cls) -> "CacheTierConfig":
        """Default config for search cache."""
        return cls(
            ttl=300,  # 5 minutes - query-dependent
            max_size=1000,  # ~1K search results
            enabled=True,
        )

    @classmethod
    def default_graph(cls) -> "CacheTierConfig":
        """Default config for graph cache."""
        return cls(
            ttl=600,  # 10 minutes - semi-static
            max_size=500,  # ~500 graph traversals
            enabled=True,
        )


@dataclass
class RedisCacheConfig:
    """Redis cache configuration."""

    # Connection settings
    redis_url: str = "redis://localhost:6379"
    key_prefix: str = "t4dm:"
    connection_timeout: int = 5  # seconds
    max_connection_attempts: int = 3

    # Fallback settings
    fallback_enabled: bool = True  # Use in-memory when Redis unavailable
    fallback_max_size: int = 10000  # Total fallback cache size

    # Tier configurations
    embedding: CacheTierConfig = field(default_factory=CacheTierConfig.default_embedding)
    search: CacheTierConfig = field(default_factory=CacheTierConfig.default_search)
    graph: CacheTierConfig = field(default_factory=CacheTierConfig.default_graph)

    # Global settings
    enabled: bool = True  # Master switch for all caching
    stats_enabled: bool = True  # Track cache statistics

    @classmethod
    def from_env(cls) -> "RedisCacheConfig":
        """
        Create config from environment variables.

        Environment variables:
            T4DM_REDIS_URL: Redis connection URL
            T4DM_REDIS_ENABLED: Enable Redis cache (true/false)
            T4DM_CACHE_FALLBACK_ENABLED: Enable fallback cache (true/false)
            T4DM_CACHE_EMBEDDING_TTL: Embedding cache TTL in seconds
            T4DM_CACHE_SEARCH_TTL: Search cache TTL in seconds
            T4DM_CACHE_GRAPH_TTL: Graph cache TTL in seconds
        """
        import os

        redis_url = os.getenv("T4DM_REDIS_URL", "redis://localhost:6379")
        enabled = os.getenv("T4DM_REDIS_ENABLED", "true").lower() == "true"
        fallback_enabled = os.getenv("T4DM_CACHE_FALLBACK_ENABLED", "true").lower() == "true"

        # Tier TTLs
        embedding_ttl = int(os.getenv("T4DM_CACHE_EMBEDDING_TTL", "3600"))
        search_ttl = int(os.getenv("T4DM_CACHE_SEARCH_TTL", "300"))
        graph_ttl = int(os.getenv("T4DM_CACHE_GRAPH_TTL", "600"))

        # Tier max sizes
        embedding_max_size = int(os.getenv("T4DM_CACHE_EMBEDDING_MAX_SIZE", "5000"))
        search_max_size = int(os.getenv("T4DM_CACHE_SEARCH_MAX_SIZE", "1000"))
        graph_max_size = int(os.getenv("T4DM_CACHE_GRAPH_MAX_SIZE", "500"))

        return cls(
            redis_url=redis_url,
            enabled=enabled,
            fallback_enabled=fallback_enabled,
            embedding=CacheTierConfig(
                ttl=embedding_ttl, max_size=embedding_max_size, enabled=enabled
            ),
            search=CacheTierConfig(ttl=search_ttl, max_size=search_max_size, enabled=enabled),
            graph=CacheTierConfig(ttl=graph_ttl, max_size=graph_max_size, enabled=enabled),
        )

    @classmethod
    def development(cls) -> "RedisCacheConfig":
        """Development environment config (permissive)."""
        return cls(
            redis_url="redis://localhost:6379",
            fallback_enabled=True,
            enabled=True,
        )

    @classmethod
    def production(cls) -> "RedisCacheConfig":
        """Production environment config (optimized)."""
        return cls(
            redis_url="redis://redis:6379",  # Docker service name
            fallback_enabled=True,  # Still allow fallback for resilience
            enabled=True,
            embedding=CacheTierConfig(
                ttl=7200,  # 2 hours in production
                max_size=10000,
                enabled=True,
            ),
            search=CacheTierConfig(
                ttl=600,  # 10 minutes in production
                max_size=2000,
                enabled=True,
            ),
            graph=CacheTierConfig(
                ttl=1800,  # 30 minutes in production
                max_size=1000,
                enabled=True,
            ),
        )

    @classmethod
    def test(cls) -> "RedisCacheConfig":
        """Test environment config (in-memory only)."""
        return cls(
            redis_url="redis://localhost:6379",
            fallback_enabled=True,
            enabled=True,  # Enable for testing cache behavior
            max_connection_attempts=1,  # Fail fast in tests
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "redis_url": self.redis_url,
            "key_prefix": self.key_prefix,
            "enabled": self.enabled,
            "fallback_enabled": self.fallback_enabled,
            "fallback_max_size": self.fallback_max_size,
            "embedding": {
                "ttl": self.embedding.ttl,
                "max_size": self.embedding.max_size,
                "enabled": self.embedding.enabled,
            },
            "search": {
                "ttl": self.search.ttl,
                "max_size": self.search.max_size,
                "enabled": self.search.enabled,
            },
            "graph": {
                "ttl": self.graph.ttl,
                "max_size": self.graph.max_size,
                "enabled": self.graph.enabled,
            },
        }
