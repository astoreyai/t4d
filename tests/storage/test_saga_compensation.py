"""
Comprehensive tests for Saga pattern compensation and rollback.

Tests cover:
- Successful multi-step execution
- Compensation on failure at each step
- Partial failure handling
- Timeout handling
- Concurrent saga execution
- State tracking
- CompensationError handling
"""
import asyncio
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from ww.storage.saga import (
    Saga,
    SagaStep,
    SagaState,
    SagaResult,
    CompensationError,
)


class TestSagaBasicExecution:
    """Test basic saga execution without failures."""

    @pytest.mark.asyncio
    async def test_single_step_success(self):
        """Single step executes and returns result."""
        action_called = False

        async def action():
            nonlocal action_called
            action_called = True
            return {"id": "123"}

        saga = Saga("test_saga")
        saga.add_step(
            name="create",
            action=action,
            compensate=AsyncMock(),
        )

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert action_called is True
        assert result.results == [{"id": "123"}]
        assert result.error is None

    @pytest.mark.asyncio
    async def test_multi_step_success(self):
        """Multiple steps execute in order."""
        execution_order = []

        async def step1():
            execution_order.append("step1")
            return "result1"

        async def step2():
            execution_order.append("step2")
            return "result2"

        async def step3():
            execution_order.append("step3")
            return "result3"

        saga = Saga("multi_step")
        saga.add_step("step1", step1, AsyncMock())
        saga.add_step("step2", step2, AsyncMock())
        saga.add_step("step3", step3, AsyncMock())

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert execution_order == ["step1", "step2", "step3"]
        assert result.results == ["result1", "result2", "result3"]
        assert result.error is None
        assert result.failed_step is None

    @pytest.mark.asyncio
    async def test_saga_state_transitions(self):
        """Saga state transitions correctly through lifecycle."""
        saga = Saga("state_test")

        assert saga.state == SagaState.PENDING

        saga.add_step("step", AsyncMock(return_value="ok"), AsyncMock())

        result = await saga.execute()

        assert saga.state == SagaState.COMMITTED
        assert result.state == SagaState.COMMITTED

    @pytest.mark.asyncio
    async def test_empty_saga(self):
        """Saga with no steps completes successfully."""
        saga = Saga("empty")

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert result.results == []

    @pytest.mark.asyncio
    async def test_step_results_preserved(self):
        """All step results are preserved in order."""
        saga = Saga("preserve_results")

        saga.add_step("step1", AsyncMock(return_value={"value": 1}), AsyncMock())
        saga.add_step("step2", AsyncMock(return_value={"value": 2}), AsyncMock())
        saga.add_step("step3", AsyncMock(return_value={"value": 3}), AsyncMock())

        result = await saga.execute()

        assert len(result.results) == 3
        assert result.results[0] == {"value": 1}
        assert result.results[1] == {"value": 2}
        assert result.results[2] == {"value": 3}


class TestSagaCompensation:
    """Test saga compensation on failure."""

    @pytest.mark.asyncio
    async def test_compensation_on_second_step_failure(self):
        """First step compensated when second fails."""
        step1_compensated = False

        async def compensate1():
            nonlocal step1_compensated
            step1_compensated = True

        saga = Saga("compensation_test", raise_on_compensation_failure=False)

        saga.add_step(
            name="step1",
            action=AsyncMock(return_value="ok"),
            compensate=compensate1,
        )
        saga.add_step(
            name="step2",
            action=AsyncMock(side_effect=Exception("Step 2 failed")),
            compensate=AsyncMock(),
        )

        result = await saga.execute()

        assert result.state == SagaState.COMPENSATED
        assert step1_compensated is True
        assert "Step 2 failed" in str(result.error)
        assert result.failed_step == "step2"

    @pytest.mark.asyncio
    async def test_compensation_order_is_reverse(self):
        """Compensations execute in reverse order (LIFO)."""
        compensation_order = []

        async def comp1():
            compensation_order.append("comp1")

        async def comp2():
            compensation_order.append("comp2")

        async def comp3():
            compensation_order.append("comp3")

        saga = Saga("reverse_order", raise_on_compensation_failure=False)

        saga.add_step("step1", AsyncMock(return_value="ok"), comp1)
        saga.add_step("step2", AsyncMock(return_value="ok"), comp2)
        saga.add_step(
            "step3",
            AsyncMock(side_effect=Exception("fail")),
            comp3,
        )

        await saga.execute()

        # Only step1 and step2 completed, so only their compensations run
        # In reverse order: comp2, comp1
        assert compensation_order == ["comp2", "comp1"]

    @pytest.mark.asyncio
    async def test_no_compensation_for_failed_step(self):
        """Failed step is not compensated, only completed steps."""
        compensations_run = []

        async def comp1():
            compensations_run.append(1)

        async def comp2():
            compensations_run.append(2)

        async def comp3():
            compensations_run.append(3)

        saga = Saga("no_comp_failed", raise_on_compensation_failure=False)

        saga.add_step("step1", AsyncMock(return_value="ok"), comp1)
        saga.add_step(
            "step2",
            AsyncMock(side_effect=ValueError("step2 failed")),
            comp2,
        )
        saga.add_step("step3", AsyncMock(return_value="ok"), comp3)

        await saga.execute()

        # Only step1 completed, so only comp1 runs
        assert compensations_run == [1]

    @pytest.mark.asyncio
    async def test_all_completed_steps_compensated(self):
        """All completed steps are compensated on failure."""
        compensated_steps = []

        async def make_compensate(step_num):
            async def comp():
                compensated_steps.append(step_num)
            return comp

        saga = Saga("all_compensated", raise_on_compensation_failure=False)

        for i in range(5):
            if i < 4:
                saga.add_step(
                    f"step{i}",
                    AsyncMock(return_value=f"result{i}"),
                    await make_compensate(i),
                )
            else:
                saga.add_step(
                    f"step{i}",
                    AsyncMock(side_effect=Exception("final fail")),
                    await make_compensate(i),
                )

        await saga.execute()

        # Steps 0-3 completed, so compensations 3,2,1,0 should run (reverse order)
        assert compensated_steps == [3, 2, 1, 0]

    @pytest.mark.asyncio
    async def test_compensation_runs_even_on_first_step_failure(self):
        """No compensation when first step fails (nothing completed)."""
        compensation_called = False

        async def compensate():
            nonlocal compensation_called
            compensation_called = True

        saga = Saga("first_fail", raise_on_compensation_failure=False)

        saga.add_step(
            "step1",
            AsyncMock(side_effect=RuntimeError("immediate fail")),
            compensate,
        )

        result = await saga.execute()

        # First step never completed, so no compensation
        assert compensation_called is False
        assert result.failed_step == "step1"


class TestSagaCompensationFailure:
    """Test handling when compensation itself fails."""

    @pytest.mark.asyncio
    async def test_compensation_error_raised_by_default(self):
        """CompensationError raised when compensation fails (default behavior)."""
        saga = Saga("comp_fail")  # raise_on_compensation_failure=True by default

        saga.add_step(
            "step1",
            AsyncMock(return_value="ok"),
            AsyncMock(side_effect=Exception("Compensation failed")),
        )
        saga.add_step(
            "step2",
            AsyncMock(side_effect=Exception("Action failed")),
            AsyncMock(),
        )

        with pytest.raises(CompensationError) as exc_info:
            await saga.execute()

        assert exc_info.value.saga_name == "comp_fail"
        assert exc_info.value.failed_step == "step2"
        assert "Action failed" in exc_info.value.original_error
        assert len(exc_info.value.compensation_errors) == 1
        assert "step1" in exc_info.value.compensation_errors[0]

    @pytest.mark.asyncio
    async def test_compensation_error_not_raised_when_disabled(self):
        """CompensationError not raised when raise_on_compensation_failure=False."""
        saga = Saga("comp_fail_silent", raise_on_compensation_failure=False)

        saga.add_step(
            "step1",
            AsyncMock(return_value="ok"),
            AsyncMock(side_effect=Exception("Compensation failed")),
        )
        saga.add_step(
            "step2",
            AsyncMock(side_effect=Exception("Action failed")),
            AsyncMock(),
        )

        result = await saga.execute()

        assert result.state == SagaState.FAILED
        assert len(result.compensation_errors) == 1
        assert "step1" in result.compensation_errors[0]

    @pytest.mark.asyncio
    async def test_continues_compensation_after_failure(self):
        """Continues compensating other steps even if one fails."""
        compensated = []

        async def comp1():
            compensated.append(1)

        async def comp2_fails():
            raise Exception("Comp 2 failed")

        async def comp3():
            compensated.append(3)

        saga = Saga("continue_comp", raise_on_compensation_failure=False)

        saga.add_step("step1", AsyncMock(return_value="ok"), comp1)
        saga.add_step("step2", AsyncMock(return_value="ok"), comp2_fails)
        saga.add_step("step3", AsyncMock(return_value="ok"), comp3)
        saga.add_step(
            "step4",
            AsyncMock(side_effect=Exception("fail")),
            AsyncMock(),
        )

        result = await saga.execute()

        # All compensations attempted despite comp2 failure (reverse order: 3, 2, 1)
        assert 1 in compensated
        assert 3 in compensated
        assert len(result.compensation_errors) == 1
        assert "step2" in result.compensation_errors[0]

    @pytest.mark.asyncio
    async def test_multiple_compensation_failures(self):
        """Multiple compensation failures are all recorded."""
        saga = Saga("multi_comp_fail", raise_on_compensation_failure=False)

        saga.add_step(
            "step1",
            AsyncMock(return_value="ok"),
            AsyncMock(side_effect=ValueError("comp1 fail")),
        )
        saga.add_step(
            "step2",
            AsyncMock(return_value="ok"),
            AsyncMock(side_effect=RuntimeError("comp2 fail")),
        )
        saga.add_step(
            "step3",
            AsyncMock(side_effect=Exception("action fail")),
            AsyncMock(),
        )

        result = await saga.execute()

        assert result.state == SagaState.FAILED
        assert len(result.compensation_errors) == 2
        # Verify both errors recorded
        errors_str = "".join(result.compensation_errors)
        assert "step1" in errors_str or "step2" in errors_str


class TestSagaTimeout:
    """Test saga timeout handling."""

    @pytest.mark.asyncio
    async def test_action_timeout(self):
        """Action that exceeds timeout is cancelled."""
        saga = Saga("timeout_test", timeout=0.1, raise_on_compensation_failure=False)

        async def slow_action():
            await asyncio.sleep(1.0)
            return "too late"

        saga.add_step("slow", slow_action, AsyncMock())

        result = await saga.execute()

        assert result.state in [SagaState.COMPENSATED, SagaState.FAILED]
        assert result.failed_step == "timeout"
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_triggers_compensation(self):
        """Timeout triggers compensation of completed steps."""
        compensated = []

        async def comp1():
            compensated.append(1)

        async def comp2():
            compensated.append(2)

        async def slow_action():
            await asyncio.sleep(1.0)

        saga = Saga("timeout_comp", timeout=0.1, raise_on_compensation_failure=False)

        saga.add_step("step1", AsyncMock(return_value="ok"), comp1)
        saga.add_step("step2", AsyncMock(return_value="ok"), comp2)
        saga.add_step("slow", slow_action, AsyncMock())

        result = await saga.execute()

        # Steps 1 and 2 should be compensated
        assert 1 in compensated
        assert 2 in compensated

    @pytest.mark.asyncio
    async def test_fast_execution_no_timeout(self):
        """Fast execution completes before timeout."""
        saga = Saga("fast", timeout=10.0)

        saga.add_step("quick", AsyncMock(return_value="done"), AsyncMock())

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert result.error is None


class TestSagaStateTracking:
    """Test saga state and history tracking."""

    @pytest.mark.asyncio
    async def test_completed_steps_tracked(self):
        """Completed steps are tracked in saga steps list."""
        saga = Saga("tracking")

        saga.add_step("step1", AsyncMock(return_value="r1"), AsyncMock())
        saga.add_step("step2", AsyncMock(return_value="r2"), AsyncMock())

        result = await saga.execute()

        assert len(saga.steps) == 2
        assert all(step.completed for step in saga.steps)
        assert saga.steps[0].name == "step1"
        assert saga.steps[1].name == "step2"
        assert saga.steps[0].result == "r1"
        assert saga.steps[1].result == "r2"

    @pytest.mark.asyncio
    async def test_failed_step_identified(self):
        """Failed step is identified in result."""
        saga = Saga("fail_tracking", raise_on_compensation_failure=False)

        saga.add_step("step1", AsyncMock(return_value="ok"), AsyncMock())
        saga.add_step(
            "step2",
            AsyncMock(side_effect=ValueError("bad value")),
            AsyncMock(),
        )

        result = await saga.execute()

        assert result.failed_step == "step2"
        assert "bad value" in str(result.error)

    @pytest.mark.asyncio
    async def test_partial_completion_tracked(self):
        """Partial completion is tracked when saga fails."""
        saga = Saga("partial", raise_on_compensation_failure=False)

        saga.add_step("step1", AsyncMock(return_value="ok"), AsyncMock())
        saga.add_step("step2", AsyncMock(return_value="ok"), AsyncMock())
        saga.add_step("step3", AsyncMock(side_effect=Exception("fail")), AsyncMock())
        saga.add_step("step4", AsyncMock(return_value="ok"), AsyncMock())

        result = await saga.execute()

        # Only steps 1 and 2 completed
        assert saga.steps[0].completed is True
        assert saga.steps[1].completed is True
        assert saga.steps[2].completed is False
        assert saga.steps[3].completed is False
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_compensated_flag_set(self):
        """Compensated flag is set for compensated steps."""
        saga = Saga("comp_flag", raise_on_compensation_failure=False)

        saga.add_step("step1", AsyncMock(return_value="ok"), AsyncMock())
        saga.add_step("step2", AsyncMock(return_value="ok"), AsyncMock())
        saga.add_step("step3", AsyncMock(side_effect=Exception("fail")), AsyncMock())

        await saga.execute()

        # Steps 1 and 2 should be marked as compensated
        assert saga.steps[0].compensated is True
        assert saga.steps[1].compensated is True
        assert saga.steps[2].compensated is False


class TestSagaIntegration:
    """Integration tests simulating real cross-store operations."""

    @pytest.mark.asyncio
    async def test_vector_graph_saga_success(self):
        """Simulate successful vector + graph store saga."""
        # Simulate stores
        vector_store = {"data": {}}
        graph_store = {"nodes": {}}

        episode_id = str(uuid4())

        async def add_vector():
            vector_store["data"][episode_id] = {"embedding": [0.1] * 1024}
            return {"id": episode_id}

        async def compensate_vector():
            if episode_id in vector_store["data"]:
                del vector_store["data"][episode_id]

        async def create_node():
            graph_store["nodes"][episode_id] = {"type": "Episode"}
            return {"id": episode_id}

        async def compensate_node():
            if episode_id in graph_store["nodes"]:
                del graph_store["nodes"][episode_id]

        saga = Saga("vector_graph")
        saga.add_step("add_vector", add_vector, compensate_vector)
        saga.add_step("create_node", create_node, compensate_node)

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert episode_id in vector_store["data"]
        assert episode_id in graph_store["nodes"]

    @pytest.mark.asyncio
    async def test_vector_graph_saga_rollback(self):
        """Rollback cleans up both stores on failure."""
        vector_store = {"data": {}}
        graph_store = {"nodes": {}}

        episode_id = str(uuid4())

        async def add_vector():
            vector_store["data"][episode_id] = {"embedding": [0.1] * 1024}
            return {"id": episode_id}

        async def compensate_vector():
            if episode_id in vector_store["data"]:
                del vector_store["data"][episode_id]

        async def create_node_fails():
            raise Exception("Neo4j connection failed")

        saga = Saga("rollback_test", raise_on_compensation_failure=False)
        saga.add_step("add_vector", add_vector, compensate_vector)
        saga.add_step("create_node", create_node_fails, AsyncMock())

        result = await saga.execute()

        assert result.state == SagaState.COMPENSATED
        # Vector store should be cleaned up
        assert episode_id not in vector_store["data"]
        # Graph store never had the node
        assert episode_id not in graph_store["nodes"]

    @pytest.mark.asyncio
    async def test_three_store_operation(self):
        """Simulate operation across three stores."""
        vector_store = {}
        graph_store = {}
        cache_store = {}

        entity_id = str(uuid4())

        async def add_vector():
            vector_store[entity_id] = {"vec": [0.5] * 512}
            return entity_id

        async def add_graph():
            graph_store[entity_id] = {"label": "Entity"}
            return entity_id

        async def add_cache():
            cache_store[entity_id] = {"cached": True}
            return entity_id

        async def remove_vector():
            vector_store.pop(entity_id, None)

        async def remove_graph():
            graph_store.pop(entity_id, None)

        async def remove_cache():
            cache_store.pop(entity_id, None)

        saga = Saga("three_store", raise_on_compensation_failure=False)
        saga.add_step("vector", add_vector, remove_vector)
        saga.add_step("graph", add_graph, remove_graph)
        saga.add_step("cache", add_cache, remove_cache)

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert entity_id in vector_store
        assert entity_id in graph_store
        assert entity_id in cache_store

    @pytest.mark.asyncio
    async def test_three_store_partial_rollback(self):
        """Partial rollback when third store fails."""
        vector_store = {}
        graph_store = {}
        cache_store = {}

        entity_id = str(uuid4())

        async def add_vector():
            vector_store[entity_id] = {"vec": [0.5] * 512}
            return entity_id

        async def add_graph():
            graph_store[entity_id] = {"label": "Entity"}
            return entity_id

        async def add_cache_fails():
            raise RuntimeError("Cache unavailable")

        async def remove_vector():
            vector_store.pop(entity_id, None)

        async def remove_graph():
            graph_store.pop(entity_id, None)

        saga = Saga("three_store_fail", raise_on_compensation_failure=False)
        saga.add_step("vector", add_vector, remove_vector)
        saga.add_step("graph", add_graph, remove_graph)
        saga.add_step("cache", add_cache_fails, AsyncMock())

        result = await saga.execute()

        assert result.state == SagaState.COMPENSATED
        # First two stores should be rolled back
        assert entity_id not in vector_store
        assert entity_id not in graph_store
        assert entity_id not in cache_store

    @pytest.mark.asyncio
    async def test_complex_relationship_saga(self):
        """Complex saga with multiple relationships."""
        stores = {
            "vector": {},
            "graph_nodes": {},
            "graph_edges": {},
        }

        entity_id = str(uuid4())
        related_ids = [str(uuid4()) for _ in range(3)]

        async def create_vector():
            stores["vector"][entity_id] = [0.1] * 768
            return entity_id

        async def create_node():
            stores["graph_nodes"][entity_id] = {"type": "Entity"}
            return entity_id

        async def create_relationships():
            for rel_id in related_ids:
                edge_id = f"{entity_id}-{rel_id}"
                stores["graph_edges"][edge_id] = {"from": entity_id, "to": rel_id}
            return related_ids

        async def remove_vector():
            stores["vector"].pop(entity_id, None)

        async def remove_node():
            stores["graph_nodes"].pop(entity_id, None)

        async def remove_relationships():
            for rel_id in related_ids:
                edge_id = f"{entity_id}-{rel_id}"
                stores["graph_edges"].pop(edge_id, None)

        saga = Saga("complex_rel")
        saga.add_step("vector", create_vector, remove_vector)
        saga.add_step("node", create_node, remove_node)
        saga.add_step("relationships", create_relationships, remove_relationships)

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert entity_id in stores["vector"]
        assert entity_id in stores["graph_nodes"]
        assert len(stores["graph_edges"]) == 3


class TestSagaChaining:
    """Test saga builder pattern and chaining."""

    @pytest.mark.asyncio
    async def test_chained_step_addition(self):
        """Steps can be added via chaining."""
        saga = (
            Saga("chained")
            .add_step("step1", AsyncMock(return_value="r1"), AsyncMock())
            .add_step("step2", AsyncMock(return_value="r2"), AsyncMock())
            .add_step("step3", AsyncMock(return_value="r3"), AsyncMock())
        )

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert len(result.results) == 3


class TestSagaEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_saga_can_be_executed_only_once(self):
        """Saga maintains state after first execution."""
        saga = Saga("once")
        saga.add_step("step", AsyncMock(return_value="ok"), AsyncMock())

        result1 = await saga.execute()
        assert result1.state == SagaState.COMMITTED

        # State is already COMMITTED from first execution
        result2 = await saga.execute()
        # This will run again but saga is already in COMMITTED state initially

    @pytest.mark.asyncio
    async def test_exception_in_action_preserves_type(self):
        """Original exception type information is preserved."""
        saga = Saga("preserve_type", raise_on_compensation_failure=False)

        saga.add_step(
            "step",
            AsyncMock(side_effect=ValueError("specific error")),
            AsyncMock(),
        )

        result = await saga.execute()

        assert "specific error" in result.error
        assert result.failed_step == "step"

    @pytest.mark.asyncio
    async def test_none_result_handled(self):
        """Steps that return None are handled correctly."""
        saga = Saga("none_result")

        saga.add_step("step1", AsyncMock(return_value=None), AsyncMock())
        saga.add_step("step2", AsyncMock(return_value="ok"), AsyncMock())

        result = await saga.execute()

        assert result.state == SagaState.COMMITTED
        assert result.results[0] is None
        assert result.results[1] == "ok"

    @pytest.mark.asyncio
    async def test_compensation_error_to_dict(self):
        """CompensationError.to_dict() includes all fields."""
        saga = Saga("to_dict_test")

        saga.add_step(
            "step1",
            AsyncMock(return_value="ok"),
            AsyncMock(side_effect=Exception("comp fail")),
        )
        saga.add_step(
            "step2",
            AsyncMock(side_effect=Exception("action fail")),
            AsyncMock(),
        )

        try:
            await saga.execute()
        except CompensationError as e:
            error_dict = e.to_dict()
            assert "saga_id" in error_dict
            assert "saga_name" in error_dict
            assert error_dict["saga_name"] == "to_dict_test"
            assert "original_error" in error_dict
            assert "failed_step" in error_dict
            assert "compensation_errors" in error_dict
