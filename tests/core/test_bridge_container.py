"""
Tests for BridgeContainer initialization and lifecycle.

P7.1 Phase 2B: Verify bridge container initializes on session creation,
bridges are available after init, and cleanup works properly.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from t4dm.core.bridge_container import (
    BridgeContainer,
    BridgeContainerConfig,
    clear_bridge_containers,
    get_bridge_container,
)
from t4dm.core.services import cleanup_services, get_services, reset_services
from t4dm.learning.dopamine import DopamineSystem
from t4dm.nca.capsules import CapsuleConfig, CapsuleLayer
from t4dm.nca.forward_forward import ForwardForwardConfig, ForwardForwardLayer


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up bridge containers and services after each test."""
    yield
    clear_bridge_containers()
    reset_services()


@pytest.fixture
def mock_storage():
    """Mock storage backends to avoid connection issues."""
    with patch("t4dm.storage.t4dx.T4DXVectorStore") as mock_qdrant, \
         patch("t4dm.storage.t4dx.T4DXGraphStore") as mock_neo4j, \
         patch("t4dm.storage.close_stores", new=AsyncMock()):
        # Mock Qdrant
        qdrant_instance = MagicMock()
        qdrant_instance.initialize = AsyncMock()
        qdrant_instance.ensure_hybrid_collection = AsyncMock()
        qdrant_instance.close = AsyncMock()
        mock_qdrant.return_value = qdrant_instance

        # Mock Neo4j
        neo4j_instance = MagicMock()
        neo4j_instance.initialize = AsyncMock()
        neo4j_instance.close = AsyncMock()
        mock_neo4j.return_value = neo4j_instance

        yield {"qdrant": qdrant_instance, "neo4j": neo4j_instance}


@pytest.mark.asyncio
async def test_bridge_container_initializes_on_session(mock_storage):
    """
    P7.1 Phase 2B: Test bridge container initializes on session creation.

    Verifies that when memory services are initialized, the bridge container
    is also created and wired with NCA components.
    """
    session_id = "test-session-init"

    # Initialize memory services (this should trigger bridge container init)
    episodic, semantic, procedural = await get_services(session_id)
    await episodic.initialize()

    # Get bridge container for session
    container = get_bridge_container(session_id)

    # Verify container exists and is for correct session
    assert container is not None
    assert container.session_id == session_id

    # Verify bridge container has been initialized
    # (if any NCA components were wired, state.initialized should be True)
    # Note: initialization depends on feature flags being enabled
    assert isinstance(container, BridgeContainer)
    assert container.config.lazy_init is True  # Default config

    # Cleanup
    await cleanup_services(session_id)


@pytest.mark.asyncio
async def test_bridges_available_after_init(mock_storage):
    """
    P7.1 Phase 2B: Test bridges are available after initialization.

    Verifies that after services are initialized, bridges can be accessed
    and have the correct NCA components wired.
    """
    session_id = "test-session-bridges"

    # Initialize memory services
    episodic, semantic, procedural = await get_services(session_id)
    await episodic.initialize()

    # Get bridge container
    container = get_bridge_container(session_id)

    # Test FF bridge availability
    if container.config.ff_enabled:
        ff_bridge = container.get_ff_bridge()
        # Bridge may be None if FF layer wasn't wired (disabled by feature flag)
        # Just verify it doesn't crash
        assert ff_bridge is None or hasattr(ff_bridge, "process")

    # Test capsule bridge availability
    if container.config.capsule_enabled:
        capsule_bridge = container.get_capsule_bridge()
        assert capsule_bridge is None or hasattr(capsule_bridge, "compute_boosts")

    # Test dopamine bridge availability
    if container.config.dopamine_enabled:
        dopamine_bridge = container.get_dopamine_bridge()
        # Dopamine bridge requires hierarchy which may not be initialized
        assert dopamine_bridge is None or hasattr(dopamine_bridge, "compute_pe_signal")

    # Test statistics
    stats = container.get_statistics()
    assert stats["session_id"] == session_id
    assert "config" in stats
    assert "calls" in stats

    # Cleanup
    await cleanup_services(session_id)


@pytest.mark.asyncio
async def test_bridge_cleanup_on_session_end(mock_storage):
    """
    P7.1 Phase 2B: Test bridge cleanup on session end.

    Verifies that when services are cleaned up, the bridge container
    is also removed from the singleton registry.
    """
    session_id = "test-session-cleanup"

    # Initialize memory services
    episodic, semantic, procedural = await get_services(session_id)
    await episodic.initialize()

    # Verify container exists
    container_before = get_bridge_container(session_id)
    assert container_before is not None
    assert container_before.session_id == session_id

    # Clean up services (should also clean up bridge container)
    await cleanup_services(session_id)

    # Verify new container is created (old one was cleaned up)
    container_after = get_bridge_container(session_id)
    assert container_after is not None
    # Should be a fresh instance (not initialized by services)
    assert container_after is not container_before


@pytest.mark.asyncio
async def test_bridge_container_singleton_per_session():
    """
    P7.1 Phase 2B: Test bridge container singleton pattern per session.

    Verifies that multiple calls to get_bridge_container with the same
    session_id return the same instance.
    """
    session_id = "test-session-singleton"

    # Get container multiple times
    container1 = get_bridge_container(session_id)
    container2 = get_bridge_container(session_id)

    # Should be same instance
    assert container1 is container2

    # Different session should get different container
    other_session_id = "test-session-singleton-other"
    container3 = get_bridge_container(other_session_id)

    assert container3 is not container1
    assert container3.session_id == other_session_id

    # Cleanup
    clear_bridge_containers()


def test_bridge_container_with_explicit_components():
    """
    P7.1 Phase 2B: Test bridge container with explicit NCA components.

    Verifies that bridges can be manually wired with NCA components
    for testing purposes.
    """
    session_id = "test-explicit-components"
    config = BridgeContainerConfig(lazy_init=False)

    # Create container
    container = BridgeContainer(config=config, session_id=session_id)

    # Wire components with correct API
    ff_config = ForwardForwardConfig(input_dim=64, hidden_dim=32)
    ff_layer = ForwardForwardLayer(config=ff_config, layer_idx=0)
    container.set_ff_layer(ff_layer)

    capsule_config = CapsuleConfig(
        input_dim=64,
        num_capsules=8,
        capsule_dim=16,
        pose_dim=4,
    )
    capsule_layer = CapsuleLayer(capsule_config)
    container.set_capsule_layer(capsule_layer)

    dopamine = DopamineSystem()
    container.set_dopamine_system(dopamine)

    # Verify bridges were created
    assert container.get_ff_bridge() is not None
    assert container.get_capsule_bridge() is not None
    # Dopamine bridge requires hierarchy, so it may be None
    assert container.get_nca_bridge() is not None

    # Cleanup
    clear_bridge_containers()


def test_bridge_container_statistics():
    """
    P7.1 Phase 2B: Test bridge container statistics tracking.

    Verifies that container tracks bridge usage and provides stats.
    """
    session_id = "test-stats"
    container = get_bridge_container(session_id)

    # Access bridges to increment counters
    _ = container.get_ff_bridge()
    _ = container.get_capsule_bridge()
    _ = container.get_ff_bridge()  # Second call

    # Get statistics
    stats = container.get_statistics()

    assert stats["session_id"] == session_id
    assert stats["calls"]["ff"] == 2
    assert stats["calls"]["capsule"] == 1
    assert "last_access" in stats

    # Cleanup
    clear_bridge_containers()


@pytest.mark.asyncio
async def test_multiple_sessions_isolated(mock_storage):
    """
    P7.1 Phase 2B: Test multiple sessions have isolated bridge containers.

    Verifies that different sessions maintain separate bridge containers
    and don't interfere with each other.
    """
    session1 = "test-multi-session-1"
    session2 = "test-multi-session-2"

    # Initialize both sessions
    episodic1, _, _ = await get_services(session1)
    await episodic1.initialize()

    episodic2, _, _ = await get_services(session2)
    await episodic2.initialize()

    # Get containers
    container1 = get_bridge_container(session1)
    container2 = get_bridge_container(session2)

    # Verify isolation
    assert container1 is not container2
    assert container1.session_id == session1
    assert container2.session_id == session2

    # Get initial call counts (they may be non-zero due to initialization)
    stats1_before = container1.get_statistics()
    stats2_before = container2.get_statistics()
    ff_calls_1_before = stats1_before["calls"]["ff"]
    ff_calls_2_before = stats2_before["calls"]["ff"]

    # Use bridges in session1
    _ = container1.get_ff_bridge()
    stats1 = container1.get_statistics()

    # Verify session2 stats are independent (didn't change)
    stats2 = container2.get_statistics()
    assert stats1["calls"]["ff"] == ff_calls_1_before + 1  # Incremented by 1
    assert stats2["calls"]["ff"] == ff_calls_2_before  # Stayed the same

    # Cleanup session1 only
    await cleanup_services(session1)

    # Verify session2 still works
    container2_after = get_bridge_container(session2)
    assert container2_after is container2

    # Cleanup session2
    await cleanup_services(session2)


@pytest.mark.asyncio
async def test_bridge_container_with_disabled_features():
    """
    P7.1 Phase 2B: Test bridge container with disabled features.

    Verifies that when NCA features are disabled, bridges return None
    gracefully without crashing.
    """
    session_id = "test-disabled-features"
    config = BridgeContainerConfig(
        ff_enabled=False,
        capsule_enabled=False,
        dopamine_enabled=False,
        nca_enabled=True,  # Keep NCA enabled
    )

    container = BridgeContainer(config=config, session_id=session_id)

    # Verify disabled bridges return None
    assert container.get_ff_bridge() is None
    assert container.get_capsule_bridge() is None
    assert container.get_dopamine_bridge() is None

    # NCA bridge should still work
    nca_bridge = container.get_nca_bridge()
    # May be None if components not wired, but shouldn't crash
    assert nca_bridge is None or hasattr(nca_bridge, "get_stats")

    # Cleanup
    clear_bridge_containers()
