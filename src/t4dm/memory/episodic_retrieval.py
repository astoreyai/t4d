"""
Episodic memory retrieval operations (search, scoring, filtering).

Handles semantic search, component scoring, and result ranking.
"""

import logging
import math
from datetime import datetime
from uuid import UUID

import numpy as np

from t4dm.core.types import Episode, Outcome, ScoredResult

logger = logging.getLogger(__name__)


class EpisodicRetrievalOps:
    """
    Retrieval operations for episodic memory.

    Handles search, scoring, and ranking of episodic memories.
    """

    def __init__(
        self,
        session_id: str,
        vector_store,
        storage_ops,
        semantic_weight: float = 0.4,
        recency_weight: float = 0.3,
        outcome_weight: float = 0.2,
        importance_weight: float = 0.1,
        recency_decay: float = 0.05,
    ):
        self.session_id = session_id
        self.vector_store = vector_store
        self.storage_ops = storage_ops
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.outcome_weight = outcome_weight
        self.importance_weight = importance_weight
        self.recency_decay = recency_decay

    def score_episode(
        self,
        episode: Episode,
        semantic_score: float,
        semantic_w: float | None = None,
        recency_w: float | None = None,
        outcome_w: float | None = None,
        importance_w: float | None = None,
    ) -> tuple[float, dict]:
        """
        Score an episode with multiple components.

        Args:
            episode: Episode to score
            semantic_score: Vector similarity score
            semantic_w: Semantic weight (defaults to instance value)
            recency_w: Recency weight (defaults to instance value)
            outcome_w: Outcome weight (defaults to instance value)
            importance_w: Importance weight (defaults to instance value)

        Returns:
            Tuple of (combined_score, components_dict)
        """
        now = datetime.now()

        # Calculate recency score
        recency_score = 1.0
        if episode.timestamp:
            time_diff = (now - episode.timestamp).total_seconds()
            time_diff_days = time_diff / 86400.0
            recency_score = math.exp(-self.recency_decay * time_diff_days)

        # Calculate outcome score
        outcome_score = {
            Outcome.SUCCESS: 1.0,
            Outcome.PARTIAL: 0.5,
            Outcome.NEUTRAL: 0.3,
            Outcome.FAILURE: 0.1,
        }.get(episode.outcome, 0.3)

        # Calculate importance score
        importance_score = episode.emotional_valence

        # Use provided weights or defaults
        sw = semantic_w if semantic_w is not None else self.semantic_weight
        rw = recency_w if recency_w is not None else self.recency_weight
        ow = outcome_w if outcome_w is not None else self.outcome_weight
        iw = importance_w if importance_w is not None else self.importance_weight

        # Combined weighted score
        combined_score = (
            sw * semantic_score +
            rw * recency_score +
            ow * outcome_score +
            iw * importance_score
        )

        components = {
            "semantic": semantic_score,
            "recency": recency_score,
            "outcome": outcome_score,
            "importance": importance_score,
        }

        return combined_score, components

    async def search_episodes(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        score_threshold: float = 0.5,
        session_filter: str | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[tuple[str, float, dict]]:
        """
        Search episodes by embedding similarity.

        Args:
            query_embedding: Query vector
            limit: Maximum results
            score_threshold: Minimum similarity threshold
            session_filter: Optional session filter
            time_start: Optional start time filter
            time_end: Optional end time filter

        Returns:
            List of (episode_id, score, payload) tuples
        """
        # Build filter dict
        filter_dict = {}

        # Determine session filter
        session_id = session_filter or (self.session_id if self.session_id != "default" else None)
        if session_id:
            filter_dict["session_id"] = session_id

        # Add time range filter if specified
        if time_start:
            filter_dict["timestamp"] = filter_dict.get("timestamp", {})
            filter_dict["timestamp"]["gte"] = time_start.timestamp()
        if time_end:
            filter_dict["timestamp"] = filter_dict.get("timestamp", {})
            filter_dict["timestamp"]["lte"] = time_end.timestamp()

        # Search in vector store
        results = await self.vector_store.search(
            collection=self.vector_store.episodes_collection,
            vector=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter_dict if filter_dict else None,
        )

        return results

    async def recall_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime,
        page_size: int = 100,
        cursor: str | None = None,
        session_filter: str | None = None,
    ) -> tuple[list[Episode], str | None]:
        """
        Retrieve episodes within a time range with pagination.

        Args:
            start_time: Start of time window (inclusive)
            end_time: End of time window (inclusive)
            page_size: Number of episodes per page (max 500)
            cursor: Pagination cursor from previous call
            session_filter: Optional session ID filter

        Returns:
            Tuple of (episodes, next_cursor)
        """
        page_size = min(page_size, 500)

        # Parse cursor
        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except ValueError:
                logger.warning(f"Invalid cursor: {cursor}, starting from 0")
                offset = 0

        # Build filter
        filter_conditions = {
            "timestamp": {
                "gte": start_time.timestamp(),
                "lte": end_time.timestamp(),
            }
        }

        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        # Scroll
        try:
            results, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                scroll_filter=filter_conditions,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            episodes = []
            for id_str, payload, _ in results:
                episode = self.storage_ops.from_payload(id_str, payload)
                episodes.append(episode)

            # Determine next cursor
            next_cursor = None
            if next_offset and len(episodes) == page_size:
                next_cursor = str(next_offset)

            logger.debug(f"Retrieved {len(episodes)} episodes for timerange")
            return episodes, next_cursor

        except Exception as e:
            logger.error(f"Error in recall_by_timerange: {e}")
            raise

    async def get_recent(
        self,
        limit: int = 20,
        session_filter: str | None = None,
    ) -> list[Episode]:
        """
        Get most recent episodes ordered by timestamp descending.

        Args:
            limit: Maximum number of episodes (max 500)
            session_filter: Optional session ID filter

        Returns:
            List of episodes ordered by most recent first
        """
        limit = min(limit, 500)

        # Build filter
        filter_conditions = {}
        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        try:
            results, _ = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                scroll_filter=filter_conditions if filter_conditions else None,
                limit=limit,
                offset=0,
                with_payload=True,
                with_vectors=False,
            )

            episodes = []
            for id_str, payload, _ in results:
                episode = self.storage_ops.from_payload(id_str, payload)
                episodes.append(episode)

            # Sort by timestamp descending
            episodes.sort(key=lambda e: e.timestamp, reverse=True)
            return episodes[:limit]

        except Exception as e:
            logger.error(f"Error in get_recent: {e}")
            raise
