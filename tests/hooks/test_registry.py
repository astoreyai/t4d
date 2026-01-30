"""Tests for hooks registry module."""

import pytest
from unittest.mock import patch

from ww.hooks.registry import (
    get_global_registry,
    clear_global_registry,
    get_registry_stats,
    get_all_registries,
    initialize_default_registries,
    REGISTRY_CORE,
    REGISTRY_EPISODIC,
    REGISTRY_SEMANTIC,
    REGISTRY_PROCEDURAL,
    REGISTRY_CONSOLIDATION,
    REGISTRY_MCP,
)
from ww.hooks.base import HookRegistry, HookPhase, HookPriority


class ConcreteHook:
    """Simple hook for testing."""

    def __init__(self, name, priority=HookPriority.NORMAL, enabled=True):
        self.name = name
        self.priority = priority
        self.enabled = enabled

    def should_execute(self, context):
        return self.enabled

    async def execute(self, context):
        return context

    def get_stats(self):
        return {"name": self.name}


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def teardown_method(self):
        """Clean up registries after each test."""
        clear_global_registry()

    def test_get_global_registry(self):
        """Test getting the global registry."""
        registry = get_global_registry("test")
        assert isinstance(registry, HookRegistry)

    def test_get_global_registry_singleton(self):
        """Test global registry is a singleton per name."""
        r1 = get_global_registry("test")
        r2 = get_global_registry("test")
        assert r1 is r2

    def test_get_different_registries(self):
        """Test different names return different registries."""
        r1 = get_global_registry("registry1")
        r2 = get_global_registry("registry2")
        assert r1 is not r2

    def test_clear_global_registry_by_name(self):
        """Test clearing a specific registry."""
        from ww.hooks.base import HookPhase

        registry = get_global_registry("test")
        hook = ConcreteHook(name="test")
        registry._hooks[HookPhase.PRE].append(hook)

        clear_global_registry("test")

        # Registry should still exist but hooks should be cleared
        registry = get_global_registry("test")
        assert len(registry._hooks[HookPhase.PRE]) == 0

    def test_clear_all_registries(self):
        """Test clearing all registries."""
        get_global_registry("reg1")
        get_global_registry("reg2")

        clear_global_registry()

        assert get_all_registries() == {}

    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        get_global_registry("test")
        stats = get_registry_stats()
        assert isinstance(stats, dict)

    def test_get_all_registries(self):
        """Test getting all registries."""
        clear_global_registry()
        get_global_registry("r1")
        get_global_registry("r2")

        registries = get_all_registries()
        assert "r1" in registries
        assert "r2" in registries


class TestRegistryConstants:
    """Tests for registry constants."""

    def test_registry_constants_exist(self):
        """Test registry constants are defined."""
        assert REGISTRY_CORE == "core"
        assert REGISTRY_EPISODIC == "episodic"
        assert REGISTRY_SEMANTIC == "semantic"
        assert REGISTRY_PROCEDURAL == "procedural"
        assert REGISTRY_CONSOLIDATION == "consolidation"
        assert REGISTRY_MCP == "mcp"

    def test_initialize_default_registries(self):
        """Test initializing default registries."""
        clear_global_registry()
        initialize_default_registries()

        registries = get_all_registries()
        assert REGISTRY_CORE in registries
        assert REGISTRY_EPISODIC in registries
        assert REGISTRY_SEMANTIC in registries
