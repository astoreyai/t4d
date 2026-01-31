"""
Dependency Injection Container for World Weaver.

Provides centralized service registration and resolution,
enabling better testability and configuration.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Container:
    """
    Simple dependency injection container.

    Supports:
    - Singleton registration
    - Factory registration
    - Instance registration
    - Scoped services (per session)
    """

    _instance: Optional["Container"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Callable] = {}
        self._instances: dict[str, Any] = {}
        self._scoped: dict[str, dict[str, Any]] = {}
        self._scoped_factories: dict[str, Callable] = {}

    @classmethod
    def get_instance(cls) -> "Container":
        """Get singleton container instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Container()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset container (for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance._singletons.clear()
                cls._instance._factories.clear()
                cls._instance._instances.clear()
                cls._instance._scoped.clear()
                cls._instance._scoped_factories.clear()
            cls._instance = None

    def register_singleton(self, key: str, factory: Callable[[], T]) -> None:
        """Register a singleton factory (lazy instantiation)."""
        self._factories[key] = factory
        logger.debug(f"Registered singleton factory: {key}")

    def register_instance(self, key: str, instance: T) -> None:
        """Register a pre-created instance."""
        self._instances[key] = instance
        logger.debug(f"Registered instance: {key}")

    def register_scoped(self, key: str, factory: Callable[[str], T]) -> None:
        """Register a scoped factory (creates per session)."""
        self._scoped_factories[key] = factory
        logger.debug(f"Registered scoped factory: {key}")

    def resolve(self, key: str, scope: str | None = None) -> Any:
        """
        Resolve a dependency.

        Args:
            key: Service key
            scope: Optional scope (e.g., session_id)

        Returns:
            Resolved service instance

        Raises:
            KeyError: If service not registered
        """
        # Check instances first (fastest)
        if key in self._instances:
            return self._instances[key]

        # Check existing singletons
        if key in self._singletons:
            return self._singletons[key]

        # Check scoped (per-session)
        if scope and key in self._scoped_factories:
            if scope not in self._scoped:
                self._scoped[scope] = {}
            if key not in self._scoped[scope]:
                factory = self._scoped_factories[key]
                self._scoped[scope][key] = factory(scope)
                logger.debug(f"Created scoped service: {key} for scope: {scope}")
            return self._scoped[scope][key]

        # Create singleton if factory exists
        if key in self._factories:
            with self._lock:
                # Double-check after lock
                if key not in self._singletons:
                    self._singletons[key] = self._factories[key]()
                    logger.debug(f"Created singleton: {key}")
            return self._singletons[key]

        raise KeyError(f"No service registered for key: {key}")

    def is_registered(self, key: str) -> bool:
        """Check if a service is registered."""
        return (
            key in self._instances
            or key in self._singletons
            or key in self._factories
            or key in self._scoped_factories
        )

    def clear_scope(self, scope: str) -> None:
        """Clear all services for a scope."""
        if scope in self._scoped:
            del self._scoped[scope]
            logger.debug(f"Cleared scope: {scope}")

    def list_registered(self) -> dict[str, str]:
        """List all registered services and their types."""
        result = {}
        for key in self._instances:
            result[key] = "instance"
        for key in self._singletons:
            result[key] = "singleton"
        for key in self._factories:
            if key not in self._singletons:
                result[key] = "singleton_factory"
        for key in self._scoped_factories:
            result[key] = "scoped"
        return result


# Convenience functions
def get_container() -> Container:
    """Get global container instance."""
    return Container.get_instance()


def configure_production() -> None:
    """Configure container for production."""
    c = get_container()

    # Register stores
    from t4dm.storage.neo4j_store import Neo4jStore
    from t4dm.storage.qdrant_store import QdrantStore

    c.register_singleton("qdrant_store", QdrantStore)
    c.register_singleton("neo4j_store", Neo4jStore)

    # Register embedding provider
    try:
        from t4dm.embedding.bge_m3 import BGEM3Embedding
        c.register_singleton("embedding_provider", BGEM3Embedding)
    except ImportError:
        logger.warning("BGEM3Embedding not available")

    logger.info("Container configured for production")


def configure_testing() -> None:
    """Configure container for testing with mocks."""
    Container.reset()
    logger.info("Container reset for testing")


# Decorator for dependency injection
def inject(key: str, scope_param: str | None = None) -> Callable[[Callable], Callable]:
    """
    Decorator to inject dependencies into functions.

    Args:
        key: Service key to inject
        scope_param: Name of function parameter containing scope

    Usage:
        @inject("embedding_provider")
        def my_function(embedding_provider: EmbeddingProvider):
            ...

        @inject("session_store", scope_param="session_id")
        def my_function(session_id: str, session_store: SessionStore):
            ...
    """
    def decorator(func: Callable) -> Callable:
        import functools
        import inspect

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            scope = None
            if scope_param and scope_param in kwargs:
                scope = kwargs[scope_param]
            elif scope_param:
                # Try to find in positional args
                try:
                    scope_idx = params.index(scope_param)
                    if scope_idx < len(args):
                        scope = args[scope_idx]
                except ValueError:
                    pass

            container = get_container()
            service = container.resolve(key, scope=scope)

            # Find which parameter to inject
            for param_name in params:
                if param_name not in kwargs:
                    param_idx = params.index(param_name)
                    if param_idx >= len(args):
                        kwargs[param_name] = service
                        break

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            scope = None
            if scope_param and scope_param in kwargs:
                scope = kwargs[scope_param]

            container = get_container()
            service = container.resolve(key, scope=scope)

            for param_name in params:
                if param_name not in kwargs:
                    param_idx = params.index(param_name)
                    if param_idx >= len(args):
                        kwargs[param_name] = service
                        break

            return await func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
