"""
Unit tests for the saga pattern implementation.

Tests cover:
- Saga state transitions and execution flow
- Step chaining and compensation
- Error handling and compensation failures
- Timeout handling
- Integration with memory stores
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from t4dm.storage.saga import (
    Saga,
    SagaState,
    SagaStep,
    SagaResult,
    MemorySaga,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_qdrant_store():
    """Create a mock Qdrant store."""
    store = AsyncMock()
    store.add = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_neo4j_store():
    """Create a mock Neo4j store."""
    store = AsyncMock()
    store.create_node = AsyncMock(return_value={"id": "test-node"})
    store.delete_node = AsyncMock(return_value=None)
    store.create_relationship = AsyncMock(return_value=None)
    store.delete_relationship = AsyncMock(return_value=None)
    store.get_relationships = AsyncMock(return_value=[])
    store.get_node = AsyncMock(return_value={"id": "test-node"})
    return store


# ============================================================================
# Saga Basic Functionality Tests
# ============================================================================

@pytest.mark.asyncio
async def test_saga_initialization():
    """Test saga initialization with default and custom parameters."""
    saga = Saga("test_saga")

    assert saga.name == "test_saga"
    assert saga.saga_id is not None
    assert len(saga.saga_id) == 8
    assert saga.state == SagaState.PENDING
    assert saga.timeout == 60.0
    assert saga.steps == []


@pytest.mark.asyncio
async def test_saga_custom_timeout():
    """Test saga initialization with custom timeout."""
    saga = Saga("test_saga", timeout=30.0)

    assert saga.timeout == 30.0


@pytest.mark.asyncio
async def test_add_step_returns_self():
    """Test that add_step returns self for chaining."""
    saga = Saga("test_saga")

    async def action():
        return "result"

    async def compensate():
        pass

    result = saga.add_step("step1", action, compensate)

    assert result is saga
    assert len(saga.steps) == 1
    assert saga.steps[0].name == "step1"


@pytest.mark.asyncio
async def test_step_chaining():
    """Test chaining multiple steps with add_step."""
    saga = Saga("test_saga")

    async def action1():
        return "result1"

    async def compensate1():
        pass

    async def action2():
        return "result2"

    async def compensate2():
        pass

    saga.add_step("step1", action1, compensate1).add_step("step2", action2, compensate2)

    assert len(saga.steps) == 2
    assert saga.steps[0].name == "step1"
    assert saga.steps[1].name == "step2"


@pytest.mark.asyncio
async def test_saga_step_dataclass():
    """Test SagaStep dataclass structure."""
    async def action():
        return "test"

    async def compensate():
        pass

    step = SagaStep(
        name="test_step",
        action=action,
        compensate=compensate,
    )

    assert step.name == "test_step"
    assert step.action is action
    assert step.compensate is compensate
    assert step.result is None
    assert step.completed is False
    assert step.compensated is False


# ============================================================================
# Saga Execution Tests (All Steps Succeed)
# ============================================================================

@pytest.mark.asyncio
async def test_saga_all_steps_succeed():
    """Test saga with all steps succeeding -> COMMITTED state."""
    saga = Saga("test_saga")

    async def step1_action():
        return "result1"

    async def step1_compensate():
        pass

    async def step2_action():
        return "result2"

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Verify final state
    assert saga.state == SagaState.COMMITTED
    assert result.state == SagaState.COMMITTED
    assert result.saga_id == saga.saga_id
    assert result.results == ["result1", "result2"]
    assert result.error is None
    assert result.failed_step is None
    assert result.compensation_errors == []

    # Verify steps were marked as completed
    assert saga.steps[0].completed is True
    assert saga.steps[1].completed is True
    assert saga.steps[0].compensated is False
    assert saga.steps[1].compensated is False


@pytest.mark.asyncio
async def test_saga_stores_step_results():
    """Test that saga stores results from each step."""
    saga = Saga("test_saga")

    async def step1_action():
        return {"data": "value1"}

    async def step1_compensate():
        pass

    async def step2_action():
        return {"data": "value2"}

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    assert len(result.results) == 2
    assert result.results[0] == {"data": "value1"}
    assert result.results[1] == {"data": "value2"}
    assert saga.steps[0].result == {"data": "value1"}
    assert saga.steps[1].result == {"data": "value2"}


@pytest.mark.asyncio
async def test_saga_empty_steps():
    """Test saga execution with no steps."""
    saga = Saga("empty_saga")

    result = await saga.execute()

    assert saga.state == SagaState.COMMITTED
    assert result.state == SagaState.COMMITTED
    assert result.results == []


# ============================================================================
# Saga Failure and Compensation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_saga_step_failure_triggers_compensation():
    """Test that step failure triggers compensation -> COMPENSATED state."""
    saga = Saga("test_saga")

    step1_compensated = False
    step2_completed = False

    async def step1_action():
        return "result1"

    async def step1_compensate():
        nonlocal step1_compensated
        step1_compensated = True

    async def step2_action():
        nonlocal step2_completed
        step2_completed = True
        raise ValueError("Step 2 failed!")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Verify compensation occurred
    assert saga.state == SagaState.COMPENSATED
    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "step2"
    assert "Step 2 failed!" in result.error
    assert len(result.compensation_errors) == 0

    # Verify compensation was called for completed steps
    assert step1_compensated is True
    assert saga.steps[0].compensated is True

    # Verify incomplete steps weren't compensated
    assert saga.steps[1].compensated is False
    assert step2_completed is True


@pytest.mark.asyncio
async def test_saga_compensation_in_reverse_order():
    """Test that compensation happens in reverse order of execution."""
    compensation_order = []

    saga = Saga("test_saga")

    async def step1_action():
        return "result1"

    async def step1_compensate():
        compensation_order.append(1)

    async def step2_action():
        return "result2"

    async def step2_compensate():
        compensation_order.append(2)

    async def step3_action():
        raise RuntimeError("Step 3 failed")

    async def step3_compensate():
        compensation_order.append(3)

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)
    saga.add_step("step3", step3_action, step3_compensate)

    result = await saga.execute()

    # Compensation should be in reverse order: 2, 1
    # (step3 didn't complete so doesn't need compensation)
    assert compensation_order == [2, 1]
    assert result.state == SagaState.COMPENSATED


@pytest.mark.asyncio
async def test_saga_compensation_failure():
    """Test saga with compensation failure -> FAILED state."""
    # Use raise_on_compensation_failure=False to get result instead of exception
    saga = Saga("test_saga", raise_on_compensation_failure=False)

    async def step1_action():
        return "result1"

    async def step1_compensate():
        raise RuntimeError("Compensation failed!")

    async def step2_action():
        raise ValueError("Step 2 failed")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Verify FAILED state due to compensation error
    assert saga.state == SagaState.FAILED
    assert result.state == SagaState.FAILED
    assert result.failed_step == "step2"
    assert len(result.compensation_errors) == 1
    assert "step1: Compensation failed!" in result.compensation_errors
    assert "Compensation failed!" in result.compensation_errors[0]


@pytest.mark.asyncio
async def test_saga_multiple_compensation_failures():
    """Test saga with multiple compensation failures."""
    # Use raise_on_compensation_failure=False to get result instead of exception
    saga = Saga("test_saga", raise_on_compensation_failure=False)

    async def step1_action():
        return "result1"

    async def step1_compensate():
        raise RuntimeError("Compensation 1 failed!")

    async def step2_action():
        return "result2"

    async def step2_compensate():
        raise RuntimeError("Compensation 2 failed!")

    async def step3_action():
        raise ValueError("Step 3 failed")

    async def step3_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)
    saga.add_step("step3", step3_action, step3_compensate)

    result = await saga.execute()

    # Verify FAILED state
    assert saga.state == SagaState.FAILED
    assert result.state == SagaState.FAILED

    # Verify both compensation errors are recorded
    assert len(result.compensation_errors) == 2
    assert any("step2" in err for err in result.compensation_errors)
    assert any("step1" in err for err in result.compensation_errors)


@pytest.mark.asyncio
async def test_saga_first_step_failure():
    """Test saga failing on first step (no compensation needed)."""
    saga = Saga("test_saga")

    async def step1_action():
        raise ValueError("First step failed!")

    async def step1_compensate():
        pass

    async def step2_action():
        return "result2"

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Verify COMPENSATED state (no compensation needed, but steps weren't completed)
    assert saga.state == SagaState.COMPENSATED
    assert result.failed_step == "step1"
    assert len(result.compensation_errors) == 0
    assert saga.steps[0].completed is False
    assert saga.steps[1].completed is False


# ============================================================================
# Saga Timeout Tests
# ============================================================================

@pytest.mark.asyncio
async def test_saga_timeout_triggers_compensation():
    """Test saga timeout triggers compensation -> COMPENSATED state."""
    saga = Saga("test_saga", timeout=0.1)

    compensated = False

    async def step1_action():
        return "result1"

    async def step1_compensate():
        nonlocal compensated
        compensated = True

    async def step2_action():
        # Sleep longer than timeout
        await asyncio.sleep(1.0)
        return "result2"

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Verify timeout was detected
    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "timeout"
    assert "timeout" in result.error.lower()
    assert compensated is True


@pytest.mark.asyncio
async def test_saga_timeout_error_message():
    """Test that timeout includes timeout duration in error."""
    saga = Saga("test_saga", timeout=0.05)

    async def step1_action():
        await asyncio.sleep(0.5)
        return "result"

    async def step1_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)

    result = await saga.execute()

    assert result.failed_step == "timeout"
    assert "0.05" in result.error


# ============================================================================
# SagaResult Tests
# ============================================================================

@pytest.mark.asyncio
async def test_saga_result_on_success():
    """Test SagaResult structure on successful saga."""
    saga = Saga("test_saga")

    async def step1_action():
        return "result1"

    async def step1_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)

    result = await saga.execute()

    assert isinstance(result, SagaResult)
    assert result.saga_id is not None
    assert result.state == SagaState.COMMITTED
    assert result.results == ["result1"]
    assert result.error is None
    assert result.failed_step is None
    assert result.compensation_errors == []


@pytest.mark.asyncio
async def test_saga_result_on_failure_with_compensation():
    """Test SagaResult contains correct error info."""
    saga = Saga("test_saga")

    async def step1_action():
        return "result1"

    async def step1_compensate():
        pass

    async def step2_action():
        raise ValueError("Custom error message")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    assert result.state == SagaState.COMPENSATED
    assert result.error == "Custom error message"
    assert result.failed_step == "step2"
    assert result.compensation_errors == []
    # Results from completed steps are preserved
    assert result.results == ["result1"]


@pytest.mark.asyncio
async def test_saga_result_on_compensation_failure():
    """Test SagaResult captures compensation errors."""
    # Use raise_on_compensation_failure=False to get result instead of exception
    saga = Saga("test_saga", raise_on_compensation_failure=False)

    async def step1_action():
        return "result1"

    async def step1_compensate():
        raise RuntimeError("Rollback failed!")

    async def step2_action():
        raise ValueError("Original failure")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    assert result.state == SagaState.FAILED
    assert result.error == "Original failure"
    assert result.failed_step == "step2"
    assert len(result.compensation_errors) == 1
    assert "step1" in result.compensation_errors[0]
    assert "Rollback failed!" in result.compensation_errors[0]


# ============================================================================
# MemorySaga Tests
# ============================================================================

@pytest.mark.asyncio
async def test_memory_saga_create_episode_success(mock_qdrant_store, mock_neo4j_store):
    """Test creating episode atomically across stores."""
    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.create_episode(
        episode_id="ep-001",
        vector=[0.1, 0.2, 0.3],
        vector_payload={"text": "episode content"},
        graph_props={"content": "episode content", "timestamp": "2025-11-27"},
        collection="episodes",
    )

    assert result.state == SagaState.COMMITTED
    assert result.error is None
    assert len(result.results) == 2

    # Verify stores were called
    mock_qdrant_store.add.assert_called_once()
    mock_neo4j_store.create_node.assert_called_once()

    # Verify no compensation occurred
    mock_qdrant_store.delete.assert_not_called()
    mock_neo4j_store.delete_node.assert_not_called()


@pytest.mark.asyncio
async def test_memory_saga_create_episode_qdrant_failure(mock_qdrant_store, mock_neo4j_store):
    """Test episode creation with Qdrant failure triggers compensation."""
    mock_qdrant_store.add.side_effect = RuntimeError("Qdrant error")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.create_episode(
        episode_id="ep-001",
        vector=[0.1, 0.2, 0.3],
        vector_payload={"text": "episode content"},
        graph_props={"content": "episode content"},
        collection="episodes",
    )

    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "add_vector"
    assert "Qdrant error" in result.error

    # Phase E1: Parallel execution calls both stores, then compensates on failure
    # Neo4j may have been called (parallel execution), but should be compensated if it succeeded
    mock_neo4j_store.create_node.assert_called_once()
    # Verify Neo4j was compensated (deleted) since Qdrant failed
    mock_neo4j_store.delete_node.assert_called_once_with(node_id="ep-001", label="Episode")


@pytest.mark.asyncio
async def test_memory_saga_create_episode_neo4j_failure(mock_qdrant_store, mock_neo4j_store):
    """Test episode creation with Neo4j failure triggers compensation."""
    mock_neo4j_store.create_node.side_effect = RuntimeError("Neo4j error")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.create_episode(
        episode_id="ep-001",
        vector=[0.1, 0.2, 0.3],
        vector_payload={"text": "episode content"},
        graph_props={"content": "episode content"},
        collection="episodes",
    )

    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "create_node"
    assert "Neo4j error" in result.error

    # Qdrant should have been compensated (deleted)
    mock_qdrant_store.delete.assert_called_once()


@pytest.mark.asyncio
async def test_memory_saga_create_entity_with_relationships(
    mock_qdrant_store, mock_neo4j_store
):
    """Test creating entity with relationships atomically."""
    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    relationships = [
        ("entity-02", "RELATED_TO", {"strength": 0.8}),
        ("entity-03", "PARENT_OF", {"role": "parent"}),
    ]

    result = await memory_saga.create_entity_with_relationships(
        entity_id="entity-01",
        vector=[0.1, 0.2, 0.3],
        vector_payload={"type": "entity"},
        graph_props={"name": "Entity 1", "type": "PERSON"},
        relationships=relationships,
        collection="entities",
    )

    assert result.state == SagaState.COMMITTED
    assert len(result.results) == 3  # vector, node, relationships

    # Verify all operations were called
    mock_qdrant_store.add.assert_called_once()
    mock_neo4j_store.create_node.assert_called_once()
    assert mock_neo4j_store.create_relationship.call_count == 2


@pytest.mark.asyncio
async def test_memory_saga_create_entity_without_relationships(
    mock_qdrant_store, mock_neo4j_store
):
    """Test creating entity without relationships (empty relationships list)."""
    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.create_entity_with_relationships(
        entity_id="entity-01",
        vector=[0.1, 0.2, 0.3],
        vector_payload={"type": "entity"},
        graph_props={"name": "Entity 1"},
        relationships=[],
        collection="entities",
    )

    assert result.state == SagaState.COMMITTED
    assert len(result.results) == 2  # vector, node (no relationships step)

    mock_qdrant_store.add.assert_called_once()
    mock_neo4j_store.create_node.assert_called_once()
    # No relationship creation since list was empty
    mock_neo4j_store.create_relationship.assert_not_called()


@pytest.mark.asyncio
async def test_memory_saga_create_entity_relationship_failure(
    mock_qdrant_store, mock_neo4j_store
):
    """Test entity creation with relationship creation failure."""
    mock_neo4j_store.create_relationship.side_effect = RuntimeError("Relationship error")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    relationships = [("entity-02", "RELATED_TO", {})]

    result = await memory_saga.create_entity_with_relationships(
        entity_id="entity-01",
        vector=[0.1, 0.2, 0.3],
        vector_payload={"type": "entity"},
        graph_props={"name": "Entity 1"},
        relationships=relationships,
        collection="entities",
    )

    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "create_relationships"

    # Compensation should clean up
    mock_qdrant_store.delete.assert_called_once()
    mock_neo4j_store.delete_node.assert_called_once()


@pytest.mark.asyncio
async def test_memory_saga_delete_memory_success(mock_qdrant_store, mock_neo4j_store):
    """Test deleting memory from both stores atomically."""
    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.delete_memory(
        memory_id="mem-001",
        memory_type="Episode",
        collection="episodes",
    )

    assert result.state == SagaState.COMMITTED

    # Verify both stores were called to delete
    mock_qdrant_store.delete.assert_called_once()
    mock_neo4j_store.delete_node.assert_called_once()


@pytest.mark.asyncio
async def test_memory_saga_delete_memory_qdrant_failure(
    mock_qdrant_store, mock_neo4j_store
):
    """Test delete memory with Qdrant failure."""
    mock_qdrant_store.delete.side_effect = RuntimeError("Qdrant delete failed")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.delete_memory(
        memory_id="mem-001",
        memory_type="Episode",
        collection="episodes",
    )

    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "delete_vector"


@pytest.mark.asyncio
async def test_memory_saga_delete_memory_neo4j_failure(
    mock_qdrant_store, mock_neo4j_store
):
    """Test delete memory with Neo4j failure triggers compensation."""
    mock_neo4j_store.delete_node.side_effect = RuntimeError("Neo4j delete failed")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.delete_memory(
        memory_id="mem-001",
        memory_type="Episode",
        collection="episodes",
    )

    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "delete_node"


@pytest.mark.asyncio
async def test_memory_saga_delete_memory_with_fetch_failure(
    mock_qdrant_store, mock_neo4j_store
):
    """Test delete memory when fetch operations fail (graceful degradation)."""
    # Fetches fail but deletion should still proceed
    mock_qdrant_store.get.side_effect = RuntimeError("Get failed")
    mock_neo4j_store.get_node.side_effect = RuntimeError("Get failed")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.delete_memory(
        memory_id="mem-001",
        memory_type="Episode",
        collection="episodes",
    )

    # Deletion should still proceed even if fetches fail
    assert result.state == SagaState.COMMITTED
    mock_qdrant_store.delete.assert_called_once()
    mock_neo4j_store.delete_node.assert_called_once()


@pytest.mark.asyncio
async def test_memory_saga_delete_memory_with_restoration(
    mock_qdrant_store, mock_neo4j_store
):
    """Test delete memory with successful fetch for restoration."""
    # Setup mocks to return data that could be restored
    mock_qdrant_store.get.return_value = [("mem-001", {"text": "content"})]
    mock_neo4j_store.get_node.return_value = {"id": "mem-001", "content": "data"}

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.delete_memory(
        memory_id="mem-001",
        memory_type="Episode",
        collection="episodes",
    )

    # Deletion should succeed normally
    assert result.state == SagaState.COMMITTED
    mock_qdrant_store.delete.assert_called_once()
    mock_neo4j_store.delete_node.assert_called_once()


@pytest.mark.asyncio
async def test_memory_saga_delete_memory_restoration_on_qdrant_failure(
    mock_qdrant_store, mock_neo4j_store
):
    """Test that vector restoration is attempted when deletion fails."""
    # Setup successful fetches
    mock_qdrant_store.get.return_value = [("mem-001", {"text": "content"})]
    mock_neo4j_store.get_node.return_value = {"id": "mem-001", "label": "Episode", "content": "data"}

    # Make Qdrant deletion fail FIRST (before Neo4j deletion)
    mock_qdrant_store.delete.side_effect = RuntimeError("Delete failed")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.delete_memory(
        memory_id="mem-001",
        memory_type="Episode",
        collection="episodes",
    )

    # Should fail on delete_vector step (first step)
    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "delete_vector"
    # No compensation is run since no steps completed successfully
    mock_neo4j_store.create_node.assert_not_called()


@pytest.mark.asyncio
async def test_memory_saga_delete_memory_no_restoration_data(
    mock_qdrant_store, mock_neo4j_store
):
    """Test delete memory restoration when no data was fetched."""
    # Fetches return empty/None
    mock_qdrant_store.get.return_value = None
    mock_neo4j_store.get_node.return_value = None

    # Make Neo4j deletion fail
    mock_neo4j_store.delete_node.side_effect = RuntimeError("Delete failed")

    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    result = await memory_saga.delete_memory(
        memory_id="mem-001",
        memory_type="Episode",
        collection="episodes",
    )

    # Should still compensate, but restoration won't do anything with None data
    assert result.state == SagaState.COMPENSATED
    assert result.failed_step == "delete_node"
    # delete_vector completed so it gets compensated, but no-op since vector_data is None
    mock_qdrant_store.delete.assert_called_once()


# ============================================================================
# State Transition Tests
# ============================================================================

@pytest.mark.asyncio
async def test_saga_state_transitions_success():
    """Test state transitions during successful execution."""
    saga = Saga("test_saga")

    state_history = [saga.state]

    async def step1_action():
        state_history.append(saga.state)
        return "result"

    async def step1_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)

    await saga.execute()

    # Should go: PENDING -> RUNNING -> COMMITTED
    assert SagaState.PENDING in state_history
    assert SagaState.RUNNING in state_history
    assert saga.state == SagaState.COMMITTED


@pytest.mark.asyncio
async def test_saga_state_transitions_with_compensation():
    """Test state transitions during failure and compensation."""
    saga = Saga("test_saga")

    async def step1_action():
        return "result"

    async def step1_compensate():
        pass

    async def step2_action():
        raise ValueError("Failed")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Should go: PENDING -> RUNNING -> COMPENSATING -> COMPENSATED
    assert result.state == SagaState.COMPENSATED


@pytest.mark.asyncio
async def test_saga_state_transitions_compensation_failure():
    """Test state transitions when compensation fails."""
    # Use raise_on_compensation_failure=False to get result instead of exception
    saga = Saga("test_saga", raise_on_compensation_failure=False)

    async def step1_action():
        return "result"

    async def step1_compensate():
        raise RuntimeError("Compensation error")

    async def step2_action():
        raise ValueError("Step failed")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Should go: PENDING -> RUNNING -> COMPENSATING -> FAILED
    assert result.state == SagaState.FAILED


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_saga_with_none_return_values():
    """Test saga handling steps that return None."""
    saga = Saga("test_saga")

    async def step1_action():
        return None

    async def step1_compensate():
        pass

    async def step2_action():
        return None

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    assert result.state == SagaState.COMMITTED
    assert result.results == [None, None]


@pytest.mark.asyncio
async def test_saga_with_complex_return_types():
    """Test saga handling various return types."""
    saga = Saga("test_saga")

    async def step1_action():
        return {"key": "value", "nested": {"deep": "data"}}

    async def step1_compensate():
        pass

    async def step2_action():
        return [1, 2, 3, {"item": "data"}]

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    assert result.state == SagaState.COMMITTED
    assert result.results[0] == {"key": "value", "nested": {"deep": "data"}}
    assert result.results[1] == [1, 2, 3, {"item": "data"}]


@pytest.mark.asyncio
async def test_saga_exception_types_preserved():
    """Test that exception messages are preserved in result."""
    saga = Saga("test_saga")

    class CustomException(Exception):
        pass

    async def step1_action():
        return "result"

    async def step1_compensate():
        pass

    async def step2_action():
        raise CustomException("Custom error details")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    assert "Custom error details" in result.error


@pytest.mark.asyncio
async def test_saga_idempotent_compensation():
    """Test that compensation doesn't run on already-compensated steps."""
    compensation_count = {"step1": 0}

    saga = Saga("test_saga")

    async def step1_action():
        return "result"

    async def step1_compensate():
        compensation_count["step1"] += 1

    async def step2_action():
        raise ValueError("Fail")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    result = await saga.execute()

    # Compensation should only run once per step
    assert compensation_count["step1"] == 1
    assert saga.steps[0].compensated is True


@pytest.mark.asyncio
async def test_saga_with_many_steps():
    """Test saga with large number of steps."""
    saga = Saga("large_saga")

    async def make_action(step_num):
        async def action():
            return f"result_{step_num}"
        return action

    async def make_compensate():
        async def compensate():
            pass
        return compensate

    # Add 10 steps
    for i in range(10):
        action = await make_action(i)
        compensate = await make_compensate()
        saga.add_step(f"step_{i}", action, compensate)

    result = await saga.execute()

    assert result.state == SagaState.COMMITTED
    assert len(result.results) == 10
    assert all(f"result_{i}" in result.results for i in range(10))


@pytest.mark.asyncio
async def test_saga_compensation_with_state_changes():
    """Test that step state is properly updated during compensation."""
    saga = Saga("test_saga")

    async def step1_action():
        return "result"

    async def step1_compensate():
        pass

    async def step2_action():
        raise ValueError("Fail")

    async def step2_compensate():
        pass

    saga.add_step("step1", step1_action, step1_compensate)
    saga.add_step("step2", step2_action, step2_compensate)

    await saga.execute()

    # Verify step states
    assert saga.steps[0].completed is True
    assert saga.steps[0].compensated is True
    assert saga.steps[1].completed is False
    assert saga.steps[1].compensated is False


# ============================================================================
# Logging and Monitoring Tests
# ============================================================================

@pytest.mark.asyncio
async def test_saga_id_uniqueness():
    """Test that saga IDs are unique."""
    sagas = [Saga(f"saga_{i}") for i in range(5)]
    saga_ids = [s.saga_id for s in sagas]

    # All IDs should be unique
    assert len(set(saga_ids)) == len(saga_ids)


@pytest.mark.asyncio
async def test_saga_id_format():
    """Test that saga ID is 8 characters (UUID4 truncated)."""
    saga = Saga("test")

    assert len(saga.saga_id) == 8
    # Should be hexadecimal
    int(saga.saga_id, 16)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow_episode_creation_and_deletion(
    mock_qdrant_store, mock_neo4j_store
):
    """Test complete workflow: create episode, then delete it."""
    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    # Create episode
    create_result = await memory_saga.create_episode(
        episode_id="ep-001",
        vector=[0.1, 0.2, 0.3],
        vector_payload={"text": "content"},
        graph_props={"content": "content"},
        collection="episodes",
    )

    assert create_result.state == SagaState.COMMITTED

    # Reset mocks for next operation
    mock_qdrant_store.reset_mock()
    mock_neo4j_store.reset_mock()

    # Delete episode
    delete_result = await memory_saga.delete_memory(
        memory_id="ep-001",
        memory_type="Episode",
        collection="episodes",
    )

    assert delete_result.state == SagaState.COMMITTED
    mock_qdrant_store.delete.assert_called_once()
    mock_neo4j_store.delete_node.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_sagas(mock_qdrant_store, mock_neo4j_store):
    """Test multiple sagas executing concurrently."""
    memory_saga = MemorySaga(mock_qdrant_store, mock_neo4j_store)

    # Create multiple sagas concurrently
    results = await asyncio.gather(
        memory_saga.create_episode(
            episode_id="ep-001",
            vector=[0.1, 0.2, 0.3],
            vector_payload={"text": "ep1"},
            graph_props={"content": "ep1"},
            collection="episodes",
        ),
        memory_saga.create_episode(
            episode_id="ep-002",
            vector=[0.4, 0.5, 0.6],
            vector_payload={"text": "ep2"},
            graph_props={"content": "ep2"},
            collection="episodes",
        ),
        memory_saga.create_episode(
            episode_id="ep-003",
            vector=[0.7, 0.8, 0.9],
            vector_payload={"text": "ep3"},
            graph_props={"content": "ep3"},
            collection="episodes",
        ),
    )

    # All should succeed
    assert all(r.state == SagaState.COMMITTED for r in results)
    assert all(r.error is None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
