"""Tests for database failure scenarios."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4


class TestVectorStoreFailures:
    """Tests for vector store failure scenarios."""

    @pytest.mark.asyncio
    async def test_vector_store_operation_with_mock_failure(self, patch_settings):
        """Test that vector store operations can fail gracefully."""
        from ww.storage.qdrant_store import QdrantStore

        # Create a store and inject a failure into an operation
        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock(side_effect=ConnectionError("Connection refused"))

        with patch("ww.storage.qdrant_store.AsyncQdrantClient", return_value=mock_client):
            store = QdrantStore()

            # Test that operation fails as expected
            with pytest.raises(ConnectionError):
                await store.add(
                    collection="test",
                    ids=["1"],
                    vectors=[[0.1] * 1024],
                    payloads=[{}],
                )

    @pytest.mark.asyncio
    async def test_vector_store_timeout_cancellation(self, patch_settings):
        """Test that vector store operations can be cancelled (timeout simulation)."""
        from ww.storage.qdrant_store import QdrantStore

        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(10)
            return None

        mock_client = AsyncMock()
        mock_client.upsert = slow_operation

        with patch("ww.storage.qdrant_store.AsyncQdrantClient", return_value=mock_client):
            store = QdrantStore()

            # Create task and cancel it to simulate timeout
            task = asyncio.create_task(store.add(
                collection="test",
                ids=["1"],
                vectors=[[0.1] * 1024],
                payloads=[{}],
            ))

            await asyncio.sleep(0.05)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task


class TestGraphStoreFailures:
    """Tests for graph store failure scenarios."""

    @pytest.mark.asyncio
    async def test_graph_store_operation_with_mock_failure(self, patch_settings):
        """Test that graph store operations can fail gracefully."""
        from ww.storage.neo4j_store import Neo4jStore

        # Mock a failing operation
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(side_effect=RuntimeError("Node creation failed"))
        mock_session.run = AsyncMock(return_value=mock_result)

        class SessionContext:
            async def __aenter__(self):
                return mock_session
            async def __aexit__(self, *args):
                pass

        mock_driver.session = MagicMock(return_value=SessionContext())

        with patch("ww.storage.neo4j_store.AsyncGraphDatabase.driver", return_value=mock_driver):
            store = Neo4jStore()

            with pytest.raises(RuntimeError):
                await store.create_node(
                    label="Episode",
                    properties={"id": "test"},
                )

    @pytest.mark.asyncio
    async def test_graph_store_query_timeout(self, patch_settings):
        """Test that graph queries can timeout."""
        from ww.storage.neo4j_store import Neo4jStore

        async def slow_query(*args, **kwargs):
            await asyncio.sleep(10)
            return []

        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_session.run = slow_query

        class SessionContext:
            async def __aenter__(self):
                return mock_session
            async def __aexit__(self, *args):
                pass

        mock_driver.session = MagicMock(return_value=SessionContext())

        with patch("ww.storage.neo4j_store.AsyncGraphDatabase.driver", return_value=mock_driver):
            store = Neo4jStore()

            # Test timeout using asyncio.wait_for
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    store.query("MATCH (n) RETURN n"),
                    timeout=0.1
                )


class TestSagaCompensation:
    """Tests for saga compensation under failures."""

    @pytest.mark.asyncio
    async def test_saga_compensation_on_second_step_failure(self):
        """Test that saga compensates when second step fails."""
        from ww.storage.saga import Saga

        step1_executed = False
        step1_compensated = False
        step2_executed = False

        async def step1_action():
            nonlocal step1_executed
            step1_executed = True
            return "step1_result"

        async def step1_compensate():
            nonlocal step1_compensated
            step1_compensated = True

        async def step2_action():
            nonlocal step2_executed
            step2_executed = True
            raise RuntimeError("Step 2 failed")

        saga = Saga("test_saga", raise_on_compensation_failure=False)
        saga.add_step("step1", step1_action, step1_compensate)
        saga.add_step("step2", step2_action, None)

        result = await saga.execute()

        assert step1_executed
        assert step1_compensated
        assert step2_executed
        assert result.state.name == "COMPENSATED"
        assert result.failed_step == "step2"

    @pytest.mark.asyncio
    async def test_saga_compensation_failure_logged(self):
        """Test that compensation failures are logged and raise CompensationError."""
        from ww.storage.saga import Saga, CompensationError

        compensations_attempted = []

        async def failing_compensate():
            compensations_attempted.append("step1")
            raise RuntimeError("Compensation failed")

        async def successful_compensate():
            compensations_attempted.append("step2")

        async def action():
            return "result"

        async def failing_action():
            raise RuntimeError("Action failed")

        saga = Saga("test_saga", raise_on_compensation_failure=True)
        saga.add_step("step1", action, failing_compensate)
        saga.add_step("step2", action, successful_compensate)
        saga.add_step("step3", failing_action, None)

        with pytest.raises(CompensationError) as exc_info:
            await saga.execute()

        # Both compensations should have been attempted
        assert "step2" in compensations_attempted
        assert "step1" in compensations_attempted

        # Verify CompensationError details
        error = exc_info.value
        assert error.failed_step == "step3"
        assert len(error.compensation_errors) == 1
        assert "step1: Compensation failed" in error.compensation_errors


class TestPartialBatchFailures:
    """Tests for partial batch operation failures."""

    @pytest.mark.asyncio
    async def test_batch_partial_success(self, deterministic_chaos, patch_settings):
        """Test batch operation with some failures."""
        from ww.storage.qdrant_store import QdrantStore

        deterministic_chaos.set_failure_rate(0.5)

        call_results = []

        async def chaotic_upsert(*args, **kwargs):
            await deterministic_chaos.maybe_fail()
            call_results.append("success")

        with patch("ww.storage.qdrant_store.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.upsert = chaotic_upsert
            mock_client.return_value = mock_instance

            store = QdrantStore()

            # Multiple attempts - some should succeed, some fail
            successes = 0
            failures = 0
            for i in range(20):
                try:
                    await store.add(
                        collection="test",
                        ids=[str(i)],
                        vectors=[[0.1] * 1024],
                        payloads=[{}],
                    )
                    successes += 1
                except Exception:
                    failures += 1

            # With 50% failure rate and deterministic seed, expect mix of successes and failures
            assert successes > 0, f"Expected some successes but got {successes}"
            assert failures > 0, f"Expected some failures but got {failures}"
