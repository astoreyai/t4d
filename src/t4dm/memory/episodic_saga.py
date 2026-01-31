"""
Saga coordination for episodic memory dual-store atomicity.

Provides transaction coordination across vector and graph stores.
"""

import logging
from uuid import UUID

from t4dm.core.types import Episode
from t4dm.storage.neo4j_store import Neo4jStore
from t4dm.storage.qdrant_store import QdrantStore
from t4dm.storage.saga import Saga, SagaResult, SagaState

logger = logging.getLogger(__name__)


class EpisodicSagaCoordinator:
    """
    Coordinates atomic operations across dual stores (vector + graph).

    Ensures that episode creation and updates maintain consistency
    across both storage backends using the saga pattern.
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: Neo4jStore,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store

    async def create_episode_saga(
        self,
        episode: Episode,
        embedding: list[float],
        payload: dict,
        graph_props: dict,
    ) -> SagaResult:
        """
        Create episode across both stores atomically.

        Args:
            episode: Episode object to create
            embedding: Vector embedding
            payload: Qdrant payload dict
            graph_props: Neo4j properties dict

        Returns:
            SagaResult with execution status

        Raises:
            RuntimeError: If saga fails
        """
        saga = Saga(f"create_episode_{episode.id}")

        # Step 1: Add to vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.vector_store.add(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
                vectors=[embedding],
                payloads=[payload],
            ),
            compensate=lambda: self.vector_store.delete(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
            ),
        )

        # Step 2: Create graph node
        saga.add_step(
            name="create_node",
            action=lambda: self.graph_store.create_node(
                label="Episode",
                properties=graph_props,
            ),
            compensate=lambda: self.graph_store.delete_node(
                node_id=str(episode.id),
                label="Episode",
            ),
        )

        # Execute saga
        result = await saga.execute()

        # Check saga result and raise on failure
        if result.state not in (SagaState.COMMITTED,):
            raise RuntimeError(
                f"Episode creation failed: {result.error} "
                f"(saga: {result.saga_id}, state: {result.state.value})"
            )

        logger.info(
            f"Created episode {episode.id} "
            f"(saga: {result.saga_id}, state: {result.state.value})"
        )

        return result

    async def promote_buffered_episode_saga(
        self,
        episode: Episode,
        embedding: list[float],
        payload: dict,
        graph_props: dict,
    ) -> SagaResult:
        """
        Promote buffered episode to permanent storage atomically.

        Args:
            episode: Episode object to promote
            embedding: Vector embedding
            payload: Qdrant payload dict
            graph_props: Neo4j properties dict

        Returns:
            SagaResult with execution status

        Raises:
            RuntimeError: If saga fails
        """
        saga = Saga(f"promote_buffered_{episode.id}")

        # Step 1: Add to vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.vector_store.add(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
                vectors=[embedding],
                payloads=[payload],
            ),
            compensate=lambda: self.vector_store.delete(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
            ),
        )

        # Step 2: Create graph node
        saga.add_step(
            name="create_node",
            action=lambda: self.graph_store.create_node(
                label="Episode",
                properties=graph_props,
            ),
            compensate=lambda: self.graph_store.delete_node(
                node_id=str(episode.id),
                label="Episode",
            ),
        )

        # Execute saga
        result = await saga.execute()

        if result.state not in (SagaState.COMMITTED,):
            raise RuntimeError(
                f"Buffered episode promotion failed: {result.error} "
                f"(saga: {result.saga_id})"
            )

        logger.info(
            f"Promoted buffered episode {episode.id} "
            f"(saga: {result.saga_id})"
        )

        return result
