"""
Global hook registry for World Weaver.

Provides centralized hook management and standard registry instances.
"""

import logging

from t4dm.hooks.base import HookRegistry

logger = logging.getLogger(__name__)

# Global registry instances
_registries: dict[str, HookRegistry] = {}


def get_global_registry(name: str = "default") -> HookRegistry:
    """
    Get or create a global hook registry.

    Args:
        name: Registry identifier (e.g., "episodic", "semantic", "consolidation")

    Returns:
        Hook registry instance
    """
    if name not in _registries:
        _registries[name] = HookRegistry(name=name)
        logger.info(f"Created global hook registry: {name}")

    return _registries[name]


def clear_global_registry(name: str | None = None) -> None:
    """
    Clear hook registry.

    Args:
        name: Registry to clear, or None for all registries
    """
    if name:
        if name in _registries:
            _registries[name].clear()
            logger.info(f"Cleared hook registry: {name}")
    else:
        for registry in _registries.values():
            registry.clear()
        _registries.clear()
        logger.info("Cleared all hook registries")


def get_all_registries() -> dict[str, HookRegistry]:
    """Get all global registries."""
    return _registries.copy()


def get_registry_stats() -> dict[str, dict]:
    """Get statistics for all registries."""
    return {
        name: registry.get_stats()
        for name, registry in _registries.items()
    }


# Standard registry names
REGISTRY_CORE = "core"
REGISTRY_EPISODIC = "episodic"
REGISTRY_SEMANTIC = "semantic"
REGISTRY_PROCEDURAL = "procedural"
REGISTRY_CONSOLIDATION = "consolidation"
REGISTRY_STORAGE_NEO4J = "storage_neo4j"
REGISTRY_STORAGE_QDRANT = "storage_qdrant"
REGISTRY_MCP = "mcp"


def initialize_default_registries() -> None:
    """
    Initialize all default registries.

    Creates registries for each World Weaver module.
    """
    registries = [
        REGISTRY_CORE,
        REGISTRY_EPISODIC,
        REGISTRY_SEMANTIC,
        REGISTRY_PROCEDURAL,
        REGISTRY_CONSOLIDATION,
        REGISTRY_STORAGE_NEO4J,
        REGISTRY_STORAGE_QDRANT,
        REGISTRY_MCP,
    ]

    for name in registries:
        get_global_registry(name)

    logger.info(f"Initialized {len(registries)} default hook registries")
