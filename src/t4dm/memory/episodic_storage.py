"""
Episodic memory storage operations (CRUD, hybrid, temporal linking).

Handles episode creation, retrieval by ID, payload conversion, and temporal sequencing.
"""

import logging
from datetime import datetime
from uuid import UUID

from t4dm.core.types import Episode, EpisodeContext, Outcome

logger = logging.getLogger(__name__)


def _validate_uuid(value: any, param_name: str) -> UUID:
    """
    DATA-006 FIX: Validate UUID parameter type.

    Args:
        value: Value to validate
        param_name: Parameter name for error message

    Returns:
        UUID instance

    Raises:
        TypeError: If value is not a UUID or valid UUID string
    """
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except (ValueError, AttributeError) as e:
            raise TypeError(
                f"{param_name} must be a valid UUID, got invalid string: {value}"
            ) from e
    raise TypeError(
        f"{param_name} must be UUID, got {type(value).__name__}"
    )


class EpisodicStorageOps:
    """
    Storage operations for episodic memory.

    Handles episode CRUD, payload conversion, and hybrid storage.
    """

    def __init__(
        self,
        session_id: str,
        vector_store,
        graph_store,
        embedding_provider,
    ):
        self.session_id = session_id
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding = embedding_provider

    def to_payload(self, episode: Episode) -> dict:
        """Convert episode to Qdrant payload."""
        now = datetime.now()
        payload = {
            "content": episode.content,
            "session_id": episode.session_id,
            "timestamp": episode.timestamp.isoformat() if episode.timestamp else now.isoformat(),
            "ingested_at": now.isoformat(),
            "emotional_valence": episode.emotional_valence,
            "outcome": episode.outcome.value,
            "stability": episode.stability,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "context": {
                "project": episode.context.project,
                "file": episode.context.file,
                "tool": episode.context.tool,
            },
        }
        # P5.2: Add temporal fields if present
        if episode.previous_episode_id:
            payload["previous_episode_id"] = str(episode.previous_episode_id)
        if episode.next_episode_id:
            payload["next_episode_id"] = str(episode.next_episode_id)
        if episode.sequence_position is not None:
            payload["sequence_position"] = episode.sequence_position
        if episode.duration_ms is not None:
            payload["duration_ms"] = episode.duration_ms
        if episode.end_timestamp:
            payload["end_timestamp"] = episode.end_timestamp.isoformat()
        return payload

    def to_graph_props(self, episode: Episode) -> dict:
        """Convert episode to graph properties."""
        return {
            "episode_id": str(episode.id),
            "sessionId": episode.session_id,  # camelCase for Neo4j
            "content": episode.content,
            "timestamp": episode.timestamp.isoformat() if episode.timestamp else None,
            "outcome": episode.outcome.value,
            "valence": episode.emotional_valence,
        }

    def from_payload(self, episode_id: str, payload: dict) -> Episode:
        """Reconstruct episode from Qdrant payload."""
        timestamp_str = payload.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else None

        # P5.2: Parse temporal fields
        prev_id_str = payload.get("previous_episode_id")
        next_id_str = payload.get("next_episode_id")
        end_ts_str = payload.get("end_timestamp")

        return Episode(
            id=UUID(episode_id),
            session_id=payload.get("session_id", "default"),
            content=payload.get("content", ""),
            embedding=None,  # Not loaded from payload
            context=EpisodeContext(**payload.get("context", {})),
            outcome=Outcome(payload.get("outcome", "neutral")),
            emotional_valence=payload.get("emotional_valence", 0.5),
            stability=payload.get("stability", 1.0),
            timestamp=timestamp,
            # P5.2: Temporal structure
            previous_episode_id=UUID(prev_id_str) if prev_id_str else None,
            next_episode_id=UUID(next_id_str) if next_id_str else None,
            sequence_position=payload.get("sequence_position"),
            duration_ms=payload.get("duration_ms"),
            end_timestamp=datetime.fromisoformat(end_ts_str) if end_ts_str else None,
        )

    async def link_episodes(self, prev_id: UUID, next_id: UUID) -> None:
        """
        P5.2: Create bidirectional temporal link between episodes.

        Updates the previous episode's next_episode_id and creates a
        SEQUENCE temporal link in the graph store.

        Args:
            prev_id: Previous episode UUID
            next_id: Next episode UUID
        """
        from t4dm.core.types import TemporalLinkType

        # Update previous episode's next_episode_id in vector store
        try:
            await self.vector_store.update_payload(
                collection=self.vector_store.episodes_collection,
                id=str(prev_id),
                payload={"next_episode_id": str(next_id)},
            )
        except Exception as e:
            logger.debug(f"Vector store update skipped (may not support partial update): {e}")

        # Create temporal link relationship in graph store
        try:
            await self.graph_store.create_relationship(
                source_id=str(prev_id),
                target_id=str(next_id),
                relation_type="TEMPORAL_SEQUENCE",
                properties={
                    "link_type": TemporalLinkType.SEQUENCE.value,
                    "strength": 1.0,
                    "created_at": datetime.now().isoformat(),
                },
            )
            logger.debug(f"Created temporal link: {prev_id} -> {next_id}")
        except Exception as e:
            logger.warning(f"Failed to create temporal link in graph: {e}")

    async def store_hybrid(self, episode: Episode, content: str) -> None:
        """
        Store episode in hybrid collection with dense + sparse vectors.

        Args:
            episode: Episode to store
            content: Original content for embedding
        """
        # Generate hybrid embeddings
        dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])

        collection = self.vector_store.episodes_collection + "_hybrid"
        await self.vector_store.add_hybrid(
            collection=collection,
            ids=[str(episode.id)],
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            payloads=[self.to_payload(episode)],
        )
        logger.debug(f"Stored hybrid vectors for episode {episode.id}")

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        """
        Get episode by ID.

        Args:
            episode_id: Episode UUID

        Returns:
            Episode or None
        """
        episode_id = _validate_uuid(episode_id, "episode_id")

        results = await self.vector_store.get(
            collection=self.vector_store.episodes_collection,
            ids=[str(episode_id)],
        )

        if results:
            id_str, payload = results[0]
            return self.from_payload(id_str, payload)

        return None

    async def batch_update_access(
        self,
        episode_ids: list[UUID],
        success: bool = True,
    ) -> int:
        """
        Update access tracking for multiple episodes in batch.

        Updates stability, access_count, and last_accessed for recalled episodes.
        Used by recall methods to track memory reinforcement via FSRS.

        Args:
            episode_ids: List of episode UUIDs to update
            success: Whether recall was successful (affects stability)

        Returns:
            Number of episodes updated
        """
        if not episode_ids:
            return 0

        try:
            # Fetch current episode payloads
            id_strings = [str(eid) for eid in episode_ids]
            results = await self.vector_store.get(
                collection=self.vector_store.episodes_collection,
                ids=id_strings,
            )

            if not results:
                return 0

            # Prepare batch updates
            now = datetime.now()
            updates = []

            for id_str, payload in results:
                # Calculate new stability
                current_stability = payload.get("stability", 1.0)
                if success:
                    # Increase stability on successful recall (bounded growth)
                    new_stability = current_stability + 0.1 * (2.0 - current_stability)
                else:
                    # Decrease stability on failed recall
                    new_stability = current_stability * 0.8

                # Prepare updated payload
                updated_payload = {
                    **payload,
                    "stability": new_stability,
                    "access_count": payload.get("access_count", 0) + 1,
                    "last_accessed": now.isoformat(),
                }

                updates.append((id_str, updated_payload))

            # Batch update payloads
            updated_count = await self.vector_store.batch_update_payloads(
                collection=self.vector_store.episodes_collection,
                updates=updates,
            )

            logger.debug(f"Batch updated {updated_count} episodes (success={success})")
            return updated_count

        except Exception as e:
            logger.warning(f"Error in batch access update: {e}")
            return 0
