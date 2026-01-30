"""
Saga Pattern for Cross-Store Transactions in World Weaver.

Provides compensation-based atomicity for operations spanning Neo4j and Qdrant.
Since these stores don't support distributed transactions, we use the saga pattern
with compensating actions to achieve eventual consistency.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class CompensationError(Exception):
    """
    Raised when saga compensation fails.

    This indicates a critical state where the system may have
    inconsistent data across stores that requires manual intervention.

    Why compensation can fail:
    - Network partition during rollback
    - Data already modified by concurrent operation
    - Store temporarily unavailable

    Recovery requires manual reconciliation using saga_id to identify affected records.
    """

    def __init__(
        self,
        saga_id: str,
        saga_name: str,
        original_error: str,
        failed_step: str,
        compensation_errors: list[str],
    ):
        self.saga_id = saga_id
        self.saga_name = saga_name
        self.original_error = original_error
        self.failed_step = failed_step
        self.compensation_errors = compensation_errors
        super().__init__(
            f"Saga[{saga_id}] '{saga_name}' compensation failed. "
            f"Original failure at '{failed_step}': {original_error}. "
            f"Compensation errors: {compensation_errors}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "saga_id": self.saga_id,
            "saga_name": self.saga_name,
            "original_error": self.original_error,
            "failed_step": self.failed_step,
            "compensation_errors": self.compensation_errors,
        }


class SagaState(Enum):
    """State of a saga execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMMITTED = "committed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


@dataclass
class SagaStep:
    """A single step in a saga with its compensation action."""
    name: str
    action: Callable[[], Coroutine[Any, Any, Any]]
    compensate: Callable[[], Coroutine[Any, Any, None]]
    result: Any = None
    completed: bool = False
    compensated: bool = False


@dataclass
class SagaResult:
    """Result of saga execution."""
    saga_id: str
    state: SagaState
    results: list[Any] = field(default_factory=list)
    error: str | None = None
    failed_step: str | None = None
    compensation_errors: list[str] = field(default_factory=list)


class Saga:
    """
    Saga orchestrator for cross-store operations.

    Example usage:
        saga = Saga("create_episode")

        # Add Qdrant step with compensation
        saga.add_step(
            name="add_vector",
            action=lambda: qdrant.add(collection, [id], [vector], [payload]),
            compensate=lambda: qdrant.delete(collection, [id]),
        )

        # Add Neo4j step with compensation
        saga.add_step(
            name="create_node",
            action=lambda: neo4j.create_node("Episode", props),
            compensate=lambda: neo4j.delete_node(id),
        )

        result = await saga.execute()
    """

    def __init__(
        self,
        name: str,
        timeout: float = 60.0,
        raise_on_compensation_failure: bool = True,
    ):
        """
        Initialize saga.

        Args:
            name: Saga name for logging
            timeout: Maximum execution time in seconds
            raise_on_compensation_failure: If True, raises CompensationError when
                compensation fails. If False, returns result with compensation_errors.
                Default True for safety (errors won't be silently ignored).
        """
        self.saga_id = str(uuid4())[:8]
        self.name = name
        self.timeout = timeout
        self.raise_on_compensation_failure = raise_on_compensation_failure
        self.steps: list[SagaStep] = []
        self.state = SagaState.PENDING

        # ATOM-P3-22: Track compensation failures for auto-recovery
        self._compensation_failures: list = []

    def add_step(
        self,
        name: str,
        action: Callable[[], Coroutine[Any, Any, Any]],
        compensate: Callable[[], Coroutine[Any, Any, None]],
    ) -> "Saga":
        """
        Add a step to the saga.

        Args:
            name: Step name for logging
            action: Async function to execute
            compensate: Async function to rollback on failure

        Returns:
            Self for chaining
        """
        self.steps.append(SagaStep(
            name=name,
            action=action,
            compensate=compensate,
        ))
        return self

    async def execute(self) -> SagaResult:
        """
        Execute all steps, compensating on failure.

        Returns:
            SagaResult with execution details
        """
        self.state = SagaState.RUNNING
        results = []
        failed_step = None
        error_msg = None

        logger.info(f"Saga[{self.saga_id}] '{self.name}' starting with {len(self.steps)} steps")

        try:
            async with asyncio.timeout(self.timeout):
                for step in self.steps:
                    try:
                        logger.debug(f"Saga[{self.saga_id}] executing step '{step.name}'")
                        step.result = await step.action()
                        step.completed = True
                        results.append(step.result)
                        logger.debug(f"Saga[{self.saga_id}] step '{step.name}' completed")
                    except Exception as e:
                        failed_step = step.name
                        error_msg = str(e)
                        logger.error(f"Saga[{self.saga_id}] step '{step.name}' failed: {e}")
                        raise

                # All steps completed successfully
                self.state = SagaState.COMMITTED
                logger.info(f"Saga[{self.saga_id}] '{self.name}' committed successfully")

                return SagaResult(
                    saga_id=self.saga_id,
                    state=self.state,
                    results=results,
                )

        except TimeoutError:
            failed_step = "timeout"
            error_msg = f"Saga exceeded {self.timeout}s timeout"
            logger.error(f"Saga[{self.saga_id}] '{self.name}' timed out")

        except Exception:
            pass  # Error already logged

        # Compensation phase
        # Execute compensating actions in reverse order (LIFO) to undo completed steps
        # This maintains consistency by unwinding the transaction backwards
        return await self._compensate(results, failed_step, error_msg)

    async def _compensate(
        self,
        results: list[Any],
        failed_step: str | None,
        error_msg: str | None,
    ) -> SagaResult:
        """
        Execute compensating actions for completed steps.

        Raises:
            CompensationError: If raise_on_compensation_failure is True and
                              compensation fails. This error contains all details
                              needed for manual intervention.
        """
        self.state = SagaState.COMPENSATING
        compensation_errors = []

        logger.info(f"Saga[{self.saga_id}] starting compensation")

        # Compensate in reverse order
        for step in reversed(self.steps):
            if step.completed and not step.compensated:
                try:
                    logger.debug(f"Saga[{self.saga_id}] compensating step '{step.name}'")
                    await step.compensate()
                    step.compensated = True
                    logger.debug(f"Saga[{self.saga_id}] step '{step.name}' compensated")
                except Exception as e:
                    comp_error = f"{step.name}: {e}"
                    compensation_errors.append(comp_error)

                    # ATOM-P3-22: Track compensation failures for dead-letter queue
                    import time
                    self._compensation_failures.append({
                        "operation": step.name,
                        "error": str(e),
                        "timestamp": time.time(),
                    })
                    if len(self._compensation_failures) > 100:
                        self._compensation_failures = self._compensation_failures[-100:]

                    logger.critical(f"Saga[{self.saga_id}] compensation failed for '{step.name}': {e}")

        if compensation_errors:
            self.state = SagaState.FAILED
            logger.error(f"Saga[{self.saga_id}] compensation incomplete: {len(compensation_errors)} errors")

            # Raise exception if configured to do so (default: True for safety)
            # Raising ensures compensation failures don't get silently ignored
            # Caller can catch CompensationError to handle manual reconciliation
            if self.raise_on_compensation_failure:
                raise CompensationError(
                    saga_id=self.saga_id,
                    saga_name=self.name,
                    original_error=error_msg or "Unknown error",
                    failed_step=failed_step or "unknown",
                    compensation_errors=compensation_errors,
                )
        else:
            self.state = SagaState.COMPENSATED
            logger.info(f"Saga[{self.saga_id}] fully compensated")

        return SagaResult(
            saga_id=self.saga_id,
            state=self.state,
            results=results,
            error=error_msg,
            failed_step=failed_step,
            compensation_errors=compensation_errors,
        )


class MemorySaga:
    """
    Pre-built sagas for common memory operations.

    Provides atomic cross-store operations for the tripartite memory system.
    """

    def __init__(self, qdrant_store, neo4j_store):
        """
        Initialize with store references.

        Args:
            qdrant_store: QdrantStore instance
            neo4j_store: Neo4jStore instance
        """
        self.qdrant = qdrant_store
        self.neo4j = neo4j_store

    async def create_episode(
        self,
        episode_id: str,
        vector: list[float],
        vector_payload: dict[str, Any],
        graph_props: dict[str, Any],
        collection: str,
    ) -> SagaResult:
        """
        Create episode in both stores atomically.

        Phase E1: Parallel execution of storage operations for performance.
        Note: Compensation remains sequential for consistency.

        Args:
            episode_id: Episode UUID string
            vector: Embedding vector
            vector_payload: Qdrant payload
            graph_props: Neo4j node properties
            collection: Qdrant collection name

        Returns:
            SagaResult with execution details
        """
        # Phase E1: Execute Qdrant and Neo4j operations in parallel
        # This reduces latency when both stores are available
        try:
            qdrant_task = self.qdrant.add(
                collection=collection,
                ids=[episode_id],
                vectors=[vector],
                payloads=[vector_payload],
            )
            neo4j_task = self.neo4j.create_node(
                label="Episode",
                properties=graph_props,
            )

            # Run in parallel with asyncio.gather
            results = await asyncio.gather(qdrant_task, neo4j_task, return_exceptions=True)

            # Check for failures
            qdrant_result, neo4j_result = results

            if isinstance(qdrant_result, Exception):
                # Qdrant failed, Neo4j may have succeeded - compensate Neo4j
                logger.error(f"Qdrant add failed: {qdrant_result}")
                failed_step = "add_vector"
                error = str(qdrant_result)
                if not isinstance(neo4j_result, Exception):
                    # Rollback Neo4j
                    await self.neo4j.delete_node(node_id=episode_id, label="Episode")
                return SagaResult(
                    saga_id="parallel_create_episode",
                    state=SagaState.COMPENSATED,
                    error=error,
                    failed_step=failed_step,
                )

            if isinstance(neo4j_result, Exception):
                # Neo4j failed, Qdrant succeeded - compensate Qdrant
                logger.error(f"Neo4j create failed: {neo4j_result}")
                failed_step = "create_node"
                error = str(neo4j_result)
                await self.qdrant.delete(collection=collection, ids=[episode_id])
                return SagaResult(
                    saga_id="parallel_create_episode",
                    state=SagaState.COMPENSATED,
                    error=error,
                    failed_step=failed_step,
                )

            # Both succeeded
            return SagaResult(
                saga_id="parallel_create_episode",
                state=SagaState.COMMITTED,
                results=[qdrant_result, neo4j_result],
            )

        except Exception as e:
            # Both operations failed (early failure before gather)
            logger.error(f"Parallel saga create_episode failed: {e}")
            return SagaResult(
                saga_id="parallel_create_episode",
                state=SagaState.COMPENSATED,
                error=str(e),
                failed_step="parallel_execution",
            )

    async def create_entity_with_relationships(
        self,
        entity_id: str,
        vector: list[float],
        vector_payload: dict[str, Any],
        graph_props: dict[str, Any],
        relationships: list[tuple[str, str, dict[str, Any]]],
        collection: str,
    ) -> SagaResult:
        """
        Create entity and its relationships atomically.

        Args:
            entity_id: Entity UUID string
            vector: Embedding vector
            vector_payload: Qdrant payload
            graph_props: Neo4j node properties
            relationships: List of (target_id, rel_type, rel_props) tuples
            collection: Qdrant collection name

        Returns:
            SagaResult with execution details
        """
        saga = Saga("create_entity_with_relationships")

        # Step 1: Add vector
        saga.add_step(
            name="add_vector",
            action=lambda: self.qdrant.add(
                collection=collection,
                ids=[entity_id],
                vectors=[vector],
                payloads=[vector_payload],
            ),
            compensate=lambda: self.qdrant.delete(
                collection=collection,
                ids=[entity_id],
            ),
        )

        # Step 2: Create node
        saga.add_step(
            name="create_node",
            action=lambda: self.neo4j.create_node(
                label="Entity",
                properties=graph_props,
            ),
            compensate=lambda: self.neo4j.delete_node(
                node_id=entity_id,
                label="Entity",
            ),
        )

        # Step 3: Create relationships (if any)
        if relationships:
            async def create_rels():
                for target_id, rel_type, rel_props in relationships:
                    await self.neo4j.create_relationship(
                        source_id=entity_id,
                        target_id=target_id,
                        rel_type=rel_type,
                        properties=rel_props,
                    )

            async def delete_rels():
                # Relationships are deleted with node, but explicit cleanup
                rels = await self.neo4j.get_relationships(entity_id)
                # Neo4j DETACH DELETE handles this, but log for completeness
                logger.debug(f"Relationships will be deleted with node: {len(rels)}")

            saga.add_step(
                name="create_relationships",
                action=create_rels,
                compensate=delete_rels,
            )

        return await saga.execute()

    async def delete_memory(
        self,
        memory_id: str,
        memory_type: str,
        collection: str,
    ) -> SagaResult:
        """
        Delete memory from both stores atomically.

        Args:
            memory_id: Memory UUID string
            memory_type: "Episode", "Entity", or "Procedure"
            collection: Qdrant collection name

        Returns:
            SagaResult with execution details
        """
        # First fetch the full data for potential restore (including vector)
        vector_data = None
        graph_data = None

        try:
            # Use get_with_vectors to preserve the actual vector for compensation
            results = await self.qdrant.get_with_vectors(collection, [memory_id])
            if results:
                # vector_data is (id, payload, vector)
                vector_data = results[0]
        except Exception as e:
            logger.debug(f"Could not fetch vector for backup: {e}")

        try:
            graph_data = await self.neo4j.get_node(memory_id, label=memory_type)
        except Exception as e:
            logger.debug(f"Could not fetch node for backup: {e}")

        saga = Saga("delete_memory")

        saga.add_step(
            name="delete_vector",
            action=lambda: self.qdrant.delete(
                collection=collection,
                ids=[memory_id],
            ),
            compensate=lambda: self._restore_vector(collection, memory_id, vector_data),
        )

        saga.add_step(
            name="delete_node",
            action=lambda: self.neo4j.delete_node(
                node_id=memory_id,
                label=memory_type,
            ),
            compensate=lambda: self._restore_node(memory_type, graph_data),
        )

        return await saga.execute()

    async def _restore_vector(
        self,
        collection: str,
        memory_id: str,
        vector_data: tuple | None,
    ) -> None:
        """Restore vector during compensation."""
        if vector_data and len(vector_data) >= 3:
            point_id, payload, vector = vector_data
            if vector is not None:
                await self.qdrant.add(
                    collection=collection,
                    ids=[memory_id],
                    vectors=[vector],
                    payloads=[payload or {}],
                )
                logger.info(f"Restored vector {memory_id} during saga compensation")
            else:
                logger.warning(f"Cannot restore vector {memory_id}: vector was None")
        elif vector_data:
            logger.warning(f"Cannot restore vector {memory_id}: incomplete data (expected 3-tuple)")

    async def _restore_node(
        self,
        label: str,
        graph_data: dict | None,
    ) -> None:
        """Restore node during compensation."""
        if graph_data:
            await self.neo4j.create_node(label=label, properties=graph_data)
