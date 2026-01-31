"""
Tests for Dependency Injection Container.

Validates singleton, scoped, and instance registration patterns.
"""

import asyncio
import pytest
import threading
from unittest.mock import Mock, MagicMock
from t4dm.core.container import (
    Container,
    get_container,
    configure_production,
    configure_testing,
    inject,
)


class DummyService:
    """Dummy service for testing."""

    def __init__(self, value: str = "test"):
        self.value = value

    def get_value(self) -> str:
        return self.value


class ScopedService:
    """Scoped service that takes a session_id."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def get_session(self) -> str:
        return self.session_id


@pytest.fixture
def container():
    """Provide a fresh container for each test."""
    Container.reset()
    c = Container.get_instance()
    yield c
    Container.reset()


def test_container_singleton(container):
    """Test that singleton returns same instance."""
    # Register singleton factory
    container.register_singleton("dummy", lambda: DummyService("singleton"))

    # Resolve multiple times
    service1 = container.resolve("dummy")
    service2 = container.resolve("dummy")

    # Should be same instance
    assert service1 is service2
    assert service1.get_value() == "singleton"


def test_container_singleton_lazy(container):
    """Test that factory is not called until resolve."""
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return DummyService("lazy")

    # Register but don't resolve
    container.register_singleton("lazy", factory)
    assert call_count == 0

    # Now resolve
    service = container.resolve("lazy")
    assert call_count == 1
    assert service.get_value() == "lazy"

    # Resolve again - should not call factory
    service2 = container.resolve("lazy")
    assert call_count == 1
    assert service is service2


def test_container_scoped(container):
    """Test that scoped returns per-scope instance."""
    # Register scoped factory
    container.register_scoped("scoped", lambda sid: ScopedService(sid))

    # Resolve for different scopes
    service1 = container.resolve("scoped", scope="session1")
    service2 = container.resolve("scoped", scope="session1")
    service3 = container.resolve("scoped", scope="session2")

    # Same scope should return same instance
    assert service1 is service2
    assert service1.get_session() == "session1"

    # Different scope should return different instance
    assert service1 is not service3
    assert service3.get_session() == "session2"


def test_container_scoped_different_scopes(container):
    """Test that different scopes get different instances."""
    container.register_scoped("scoped", lambda sid: ScopedService(sid))

    sessions = ["s1", "s2", "s3"]
    services = [container.resolve("scoped", scope=s) for s in sessions]

    # Each session should have its own instance
    for i, service in enumerate(services):
        assert service.get_session() == sessions[i]

    # All should be different objects
    for i in range(len(services)):
        for j in range(i + 1, len(services)):
            assert services[i] is not services[j]


def test_container_instance(container):
    """Test that pre-created instance is returned."""
    # Create instance
    instance = DummyService("prebuilt")

    # Register instance
    container.register_instance("instance", instance)

    # Resolve should return same instance
    resolved = container.resolve("instance")
    assert resolved is instance
    assert resolved.get_value() == "prebuilt"


def test_container_reset(container):
    """Test that reset clears all services."""
    # Register services
    container.register_singleton("sing", lambda: DummyService("s"))
    container.register_instance("inst", DummyService("i"))
    container.register_scoped("scoped", lambda s: ScopedService(s))

    # Resolve to create instances
    container.resolve("sing")
    container.resolve("scoped", scope="test")

    # Reset
    Container.reset()

    # Get new container
    new_container = Container.get_instance()

    # Should not find old services
    with pytest.raises(KeyError):
        new_container.resolve("sing")

    with pytest.raises(KeyError):
        new_container.resolve("inst")

    with pytest.raises(KeyError):
        new_container.resolve("scoped", scope="test")


def test_container_key_error(container):
    """Test KeyError for unregistered service."""
    with pytest.raises(KeyError) as exc_info:
        container.resolve("nonexistent")

    assert "No service registered for key: nonexistent" in str(exc_info.value)


def test_container_is_registered(container):
    """Test is_registered method."""
    # Nothing registered yet
    assert not container.is_registered("test")

    # Register singleton factory
    container.register_singleton("sing", lambda: DummyService())
    assert container.is_registered("sing")

    # Register instance
    container.register_instance("inst", DummyService())
    assert container.is_registered("inst")

    # Register scoped
    container.register_scoped("scoped", lambda s: ScopedService(s))
    assert container.is_registered("scoped")

    # Resolve singleton (creates it)
    container.resolve("sing")
    assert container.is_registered("sing")


def test_container_list_registered(container):
    """Test list_registered shows all services."""
    # Register different types
    container.register_singleton("sing1", lambda: DummyService("s1"))
    container.register_instance("inst1", DummyService("i1"))
    container.register_scoped("scoped1", lambda s: ScopedService(s))

    # List before resolving
    registered = container.list_registered()
    assert registered["sing1"] == "singleton_factory"
    assert registered["inst1"] == "instance"
    assert registered["scoped1"] == "scoped"

    # Resolve singleton
    container.resolve("sing1")

    # List after resolving
    registered = container.list_registered()
    assert registered["sing1"] == "singleton"  # Now created
    assert registered["inst1"] == "instance"
    assert registered["scoped1"] == "scoped"


def test_container_clear_scope(container):
    """Test clear_scope removes scope-specific services."""
    container.register_scoped("scoped", lambda s: ScopedService(s))

    # Create services for multiple scopes
    s1 = container.resolve("scoped", scope="session1")
    s2 = container.resolve("scoped", scope="session2")

    assert s1.get_session() == "session1"
    assert s2.get_session() == "session2"

    # Clear session1 scope
    container.clear_scope("session1")

    # session2 should still work and return same instance
    s2_again = container.resolve("scoped", scope="session2")
    assert s2_again is s2

    # session1 should create new instance
    s1_new = container.resolve("scoped", scope="session1")
    assert s1_new is not s1
    assert s1_new.get_session() == "session1"


def test_container_thread_safety(container):
    """Test that concurrent access is safe."""
    results = []
    errors = []

    def worker(worker_id: int):
        try:
            # Each worker registers and resolves
            key = f"service_{worker_id}"
            container.register_singleton(key, lambda: DummyService(f"w{worker_id}"))
            service = container.resolve(key)
            results.append((worker_id, service.get_value()))
        except Exception as e:
            errors.append((worker_id, e))

    # Create threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

    # Start all threads
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Check results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 10

    # Each worker should have its own service
    values = {r[1] for r in results}
    assert len(values) == 10  # All unique


def test_inject_decorator(container):
    """Test inject decorator works correctly."""
    # Register service
    container.register_singleton("test_service", lambda: DummyService("injected"))

    # Define function with injection
    @inject("test_service")
    def my_function(test_service: DummyService):
        return test_service.get_value()

    # Call function - service should be injected
    result = my_function()
    assert result == "injected"


def test_inject_decorator_with_scope(container):
    """Test inject decorator with scoped service."""
    # Register scoped service
    container.register_scoped("scoped_service", lambda s: ScopedService(s))

    # Define function with scoped injection
    @inject("scoped_service", scope_param="session_id")
    def my_function(session_id: str, scoped_service: ScopedService):
        return scoped_service.get_session()

    # Call with session_id
    result = my_function(session_id="test_session")
    assert result == "test_session"


@pytest.mark.asyncio
async def test_inject_decorator_async(container):
    """Test inject decorator with async functions."""
    # Register service
    container.register_singleton("async_service", lambda: DummyService("async"))

    # Define async function with injection
    @inject("async_service")
    async def my_async_function(async_service: DummyService):
        await asyncio.sleep(0.01)
        return async_service.get_value()

    # Call async function
    result = await my_async_function()
    assert result == "async"


def test_configure_production(container):
    """Test production configuration registers services."""
    # Reset to ensure clean state
    Container.reset()

    # Configure for production
    configure_production()

    # Get container
    c = get_container()

    # Check services are registered
    assert c.is_registered("qdrant_store")
    assert c.is_registered("neo4j_store")
    # Note: embedding_provider may not be registered if imports fail


def test_configure_testing():
    """Test testing configuration resets container."""
    # Create some state
    c = get_container()
    c.register_singleton("test", lambda: DummyService())
    c.resolve("test")

    # Configure for testing (should reset)
    configure_testing()

    # Get new container
    new_c = get_container()

    # Old service should be gone
    with pytest.raises(KeyError):
        new_c.resolve("test")


def test_get_container_singleton():
    """Test get_container returns singleton instance."""
    Container.reset()

    c1 = get_container()
    c2 = get_container()

    assert c1 is c2


def test_container_resolution_priority(container):
    """Test that instance > singleton > factory in resolution."""
    # Register factory
    container.register_singleton("test", lambda: DummyService("factory"))

    # Resolve once (creates singleton)
    s1 = container.resolve("test")
    assert s1.get_value() == "factory"

    # Override with instance
    instance = DummyService("instance")
    container.register_instance("test", instance)

    # Should now resolve to instance
    s2 = container.resolve("test")
    assert s2 is instance
    assert s2.get_value() == "instance"


def test_scoped_without_scope_raises(container):
    """Test that scoped service without scope parameter raises KeyError."""
    container.register_scoped("scoped", lambda s: ScopedService(s))

    # Try to resolve without scope - should fail
    with pytest.raises(KeyError):
        container.resolve("scoped")


def test_concurrent_singleton_creation(container):
    """Test that singleton is only created once even with concurrent access."""
    creation_count = 0
    creation_lock = threading.Lock()

    def factory():
        nonlocal creation_count
        with creation_lock:
            creation_count += 1
        # Simulate some work
        import time
        time.sleep(0.01)
        return DummyService(f"concurrent_{creation_count}")

    container.register_singleton("concurrent", factory)

    results = []

    def worker():
        service = container.resolve("concurrent")
        results.append(service)

    # Create multiple threads that try to resolve simultaneously
    threads = [threading.Thread(target=worker) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Should have created exactly once
    assert creation_count == 1

    # All results should be the same instance
    assert len(results) == 10
    first = results[0]
    for r in results:
        assert r is first


def test_inject_with_explicit_args(container):
    """Test that inject doesn't override explicit arguments."""
    container.register_singleton("service", lambda: DummyService("default"))

    @inject("service")
    def my_function(service: DummyService):
        return service.get_value()

    # Explicit argument should be used
    explicit = DummyService("explicit")
    result = my_function(service=explicit)
    assert result == "explicit"


def test_multiple_scopes_independent(container):
    """Test that multiple scoped services are independent per scope."""
    container.register_scoped("service_a", lambda s: ScopedService(f"a_{s}"))
    container.register_scoped("service_b", lambda s: ScopedService(f"b_{s}"))

    # Resolve both for same scope
    a1 = container.resolve("service_a", scope="scope1")
    b1 = container.resolve("service_b", scope="scope1")

    # Resolve both for different scope
    a2 = container.resolve("service_a", scope="scope2")
    b2 = container.resolve("service_b", scope="scope2")

    # Check isolation
    assert a1.get_session() == "a_scope1"
    assert b1.get_session() == "b_scope1"
    assert a2.get_session() == "a_scope2"
    assert b2.get_session() == "b_scope2"

    # Same service, same scope = same instance
    a1_again = container.resolve("service_a", scope="scope1")
    assert a1 is a1_again


@pytest.mark.asyncio
async def test_inject_decorator_async_with_scope(container):
    """Test inject decorator with async functions and scoped services."""
    # Register scoped service
    container.register_scoped("scoped_async", lambda s: ScopedService(s))

    # Define async function with scoped injection
    @inject("scoped_async", scope_param="session_id")
    async def my_async_function(session_id: str, scoped_async: ScopedService):
        await asyncio.sleep(0.01)
        return scoped_async.get_session()

    # Call async function
    result = await my_async_function(session_id="async_session")
    assert result == "async_session"
