"""
Retrieval Feedback Loop System for World Weaver.

Phase 1C: Implements implicit feedback collection from retrieval outcomes
to enable continuous learning from user interactions.

Biological Basis:
- The brain learns from outcomes without explicit reward signals
- Attention (dwell time) correlates with relevance/engagement
- Selection behavior (clicks) indicates preference
- The hippocampus tracks which retrieved memories were actually useful

This system collects implicit feedback signals:
1. Click-through: User selected this result (positive signal)
2. Dwell time: User spent time reading/using this result
3. Skip: User saw but didn't select (weak negative signal)
4. Query refinement: User refined query (result wasn't sufficient)

The collected feedback feeds into:
- Three-factor learning rule (dopamine RPE)
- Self-supervised credit assignment
- Adapter/scorer training

References:
- Joachims et al. (2007): Evaluating retrieval performance using clickthrough
- Craswell et al. (2008): An experimental comparison of click position-bias models
- Chapelle & Zhang (2009): A dynamic bayesian network click model
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

logger = logging.getLogger(__name__)

# Security limits
MAX_RESULTS_PER_QUERY = 100
MAX_DWELL_TIME_SECONDS = 3600.0  # 1 hour cap
MAX_BATCH_SIZE = 1000
MAX_QUERY_ID_LENGTH = 64
MAX_RESULT_ID_LENGTH = 256


@dataclass
class RetrievalFeedback:
    """
    Single feedback event from a retrieval outcome.

    Captures implicit signals about result relevance based on user behavior.
    """
    feedback_id: UUID = field(default_factory=uuid4)
    query_id: str = ""
    result_id: str = ""

    # Relevance signal [0, 1] computed from implicit feedback
    relevance: float = 0.0

    # Component signals that contributed to relevance
    clicked: bool = False
    dwell_time: float = 0.0  # Seconds spent on result
    position: int = 0  # Rank position in results (0-indexed)

    # Confidence in this feedback signal [0, 1]
    confidence: float = 0.5

    # Temporal context
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": str(self.feedback_id),
            "query_id": self.query_id,
            "result_id": self.result_id,
            "relevance": self.relevance,
            "clicked": self.clicked,
            "dwell_time": self.dwell_time,
            "position": self.position,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalFeedback:
        """Reconstruct from dictionary."""
        return cls(
            feedback_id=UUID(data["feedback_id"]),
            query_id=data["query_id"],
            result_id=data["result_id"],
            relevance=data["relevance"],
            clicked=data["clicked"],
            dwell_time=data["dwell_time"],
            position=data["position"],
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data.get("session_id", "default"),
        )


@dataclass
class RetrievalOutcome:
    """
    Complete retrieval outcome for a single query.

    Used internally to track what happened after retrieval.
    """
    query_id: str
    query_embedding: np.ndarray | None = None
    result_ids: list[str] = field(default_factory=list)
    result_scores: dict[str, float] = field(default_factory=dict)
    clicked: list[str] = field(default_factory=list)
    dwell_times: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = "default"
    finalized: bool = False


class RetrievalFeedbackCollector:
    """
    Collect implicit feedback from retrieval outcomes.

    This system tracks retrieval events and user interactions to derive
    implicit relevance signals without requiring explicit user feedback.

    Usage:
        collector = RetrievalFeedbackCollector()

        # After retrieval, record results shown
        query_id = collector.start_retrieval(
            query_embedding=embedding,
            results=["mem_1", "mem_2", "mem_3"],
            scores={"mem_1": 0.9, "mem_2": 0.7, "mem_3": 0.5}
        )

        # As user interacts, record behavior
        collector.record_click(query_id, "mem_1")
        collector.record_dwell(query_id, "mem_1", dwell_time=30.0)

        # Finalize to compute feedback
        feedbacks = collector.finalize_retrieval(query_id)

        # Get training batch
        batch = collector.get_training_batch(batch_size=32)

    Biological Analogy:
    - start_retrieval = hippocampal pattern completion activated memories
    - record_click = attention selected this memory for processing
    - record_dwell = working memory held this content
    - finalize = consolidation checkpoint, compute credit
    """

    def __init__(
        self,
        # Relevance computation parameters
        click_weight: float = 0.6,
        dwell_weight: float = 0.3,
        position_weight: float = 0.1,
        # Dwell time parameters
        min_dwell_seconds: float = 1.0,  # Below this = too short to count
        optimal_dwell_seconds: float = 30.0,  # Optimal engagement time
        max_dwell_seconds: float = 300.0,  # Beyond this = diminishing returns
        # Position bias correction
        position_decay: float = 0.9,  # Decay factor per position
        # Storage
        db_path: str | Path | None = None,
        # Limits
        max_pending: int = 1000,
        retention_days: int = 30,
    ):
        """
        Initialize feedback collector.

        Args:
            click_weight: Weight for click signal in relevance
            dwell_weight: Weight for dwell time in relevance
            position_weight: Weight for position bias correction
            min_dwell_seconds: Minimum dwell to count as engagement
            optimal_dwell_seconds: Dwell time for maximum signal
            max_dwell_seconds: Dwell time cap for normalization
            position_decay: Position bias decay factor
            db_path: Path to SQLite database for persistence
            max_pending: Maximum pending (unfinalized) retrievals
            retention_days: Days to retain feedback data
        """
        # Validate weights
        total_weight = click_weight + dwell_weight + position_weight
        if total_weight <= 0:
            raise ValueError("Weights must sum to positive value")

        # Normalize weights
        self.click_weight = click_weight / total_weight
        self.dwell_weight = dwell_weight / total_weight
        self.position_weight = position_weight / total_weight

        self.min_dwell_seconds = min_dwell_seconds
        self.optimal_dwell_seconds = optimal_dwell_seconds
        self.max_dwell_seconds = max_dwell_seconds
        self.position_decay = position_decay
        self.max_pending = max_pending
        self.retention_days = retention_days

        # Pending retrievals (not yet finalized)
        self._pending: dict[str, RetrievalOutcome] = {}
        self._lock = threading.RLock()

        # Statistics
        self._total_retrievals = 0
        self._total_feedbacks = 0
        self._total_clicks = 0

        # SQLite persistence
        self._db_path = Path(db_path) if db_path else None
        self._conn: sqlite3.Connection | None = None
        if self._db_path:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for feedback persistence."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                result_id TEXT NOT NULL,
                relevance REAL NOT NULL,
                clicked INTEGER NOT NULL,
                dwell_time REAL NOT NULL,
                position INTEGER NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_query
            ON feedback(query_id)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_result
            ON feedback(result_id)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
            ON feedback(timestamp)
        """)

        self._conn.commit()

    def start_retrieval(
        self,
        results: list[str],
        scores: dict[str, float] | None = None,
        query_embedding: np.ndarray | None = None,
        query_id: str | None = None,
        session_id: str = "default",
    ) -> str:
        """
        Start tracking a retrieval event.

        Call this when results are shown to user, before interaction.

        Args:
            results: List of result IDs in rank order
            scores: Optional retrieval scores for each result
            query_embedding: Optional query vector for later training
            query_id: Optional custom query ID (generated if None)
            session_id: Session identifier

        Returns:
            Query ID for subsequent interaction recording
        """
        # Validate inputs
        if len(results) > MAX_RESULTS_PER_QUERY:
            results = results[:MAX_RESULTS_PER_QUERY]
            logger.warning(f"Truncated results to {MAX_RESULTS_PER_QUERY}")

        # Generate query ID if not provided
        if query_id is None:
            query_id = hashlib.sha256(
                f"{datetime.now().isoformat()}-{uuid4()}".encode()
            ).hexdigest()[:16]

        if len(query_id) > MAX_QUERY_ID_LENGTH:
            query_id = query_id[:MAX_QUERY_ID_LENGTH]

        with self._lock:
            # Enforce max pending limit
            if len(self._pending) >= self.max_pending:
                self._evict_oldest_pending()

            outcome = RetrievalOutcome(
                query_id=query_id,
                query_embedding=query_embedding,
                result_ids=results,
                result_scores=scores or {},
                session_id=session_id,
            )

            self._pending[query_id] = outcome
            self._total_retrievals += 1

        logger.debug(f"Started tracking retrieval {query_id} with {len(results)} results")
        return query_id

    def record_click(self, query_id: str, result_id: str) -> bool:
        """
        Record that user clicked/selected a result.

        Args:
            query_id: Query ID from start_retrieval
            result_id: ID of clicked result

        Returns:
            True if recorded, False if query not found
        """
        if len(result_id) > MAX_RESULT_ID_LENGTH:
            result_id = result_id[:MAX_RESULT_ID_LENGTH]

        with self._lock:
            if query_id not in self._pending:
                logger.debug(f"Query {query_id} not found for click recording")
                return False

            outcome = self._pending[query_id]
            if result_id not in outcome.clicked:
                outcome.clicked.append(result_id)
                self._total_clicks += 1

        return True

    def record_dwell(
        self,
        query_id: str,
        result_id: str,
        dwell_time: float
    ) -> bool:
        """
        Record dwell time on a result.

        Args:
            query_id: Query ID from start_retrieval
            result_id: ID of result
            dwell_time: Time in seconds spent on result

        Returns:
            True if recorded, False if query not found
        """
        if len(result_id) > MAX_RESULT_ID_LENGTH:
            result_id = result_id[:MAX_RESULT_ID_LENGTH]

        # Validate dwell time
        if not np.isfinite(dwell_time):
            logger.warning(f"Invalid dwell time: {dwell_time}")
            return False

        dwell_time = min(max(0.0, dwell_time), MAX_DWELL_TIME_SECONDS)

        with self._lock:
            if query_id not in self._pending:
                return False

            outcome = self._pending[query_id]
            # Accumulate dwell time (user might view multiple times)
            current = outcome.dwell_times.get(result_id, 0.0)
            outcome.dwell_times[result_id] = current + dwell_time

        return True

    def record_retrieval(
        self,
        query_id: str,
        results: list[str],
        clicked: list[str],
        dwell_times: dict[str, float],
        scores: dict[str, float] | None = None,
        query_embedding: np.ndarray | None = None,
        session_id: str = "default",
    ) -> list[RetrievalFeedback]:
        """
        Record a complete retrieval with all interaction data at once.

        Convenience method that combines start_retrieval, record_click,
        record_dwell, and finalize_retrieval.

        Args:
            query_id: Unique identifier for this query
            results: List of result IDs in rank order
            clicked: List of result IDs that were clicked
            dwell_times: Mapping of result_id to dwell time in seconds
            scores: Optional retrieval scores
            query_embedding: Optional query vector
            session_id: Session identifier

        Returns:
            List of computed feedback signals
        """
        # Start retrieval
        self.start_retrieval(
            results=results,
            scores=scores,
            query_embedding=query_embedding,
            query_id=query_id,
            session_id=session_id,
        )

        # Record clicks
        for result_id in clicked:
            self.record_click(query_id, result_id)

        # Record dwell times
        for result_id, dwell in dwell_times.items():
            self.record_dwell(query_id, result_id, dwell)

        # Finalize and return feedback
        return self.finalize_retrieval(query_id)

    def finalize_retrieval(self, query_id: str) -> list[RetrievalFeedback]:
        """
        Finalize a retrieval and compute feedback signals.

        Call this when user interaction with results is complete.

        Args:
            query_id: Query ID to finalize

        Returns:
            List of RetrievalFeedback for each result
        """
        with self._lock:
            if query_id not in self._pending:
                logger.debug(f"Query {query_id} not found for finalization")
                return []

            outcome = self._pending.pop(query_id)

        feedbacks = []

        for position, result_id in enumerate(outcome.result_ids):
            relevance = self.compute_relevance(
                result_id=result_id,
                position=position,
                clicked=outcome.clicked,
                dwell_times=outcome.dwell_times,
            )

            confidence = self._compute_confidence(
                result_id=result_id,
                clicked=outcome.clicked,
                dwell_times=outcome.dwell_times,
            )

            feedback = RetrievalFeedback(
                query_id=query_id,
                result_id=result_id,
                relevance=relevance,
                clicked=result_id in outcome.clicked,
                dwell_time=outcome.dwell_times.get(result_id, 0.0),
                position=position,
                confidence=confidence,
                session_id=outcome.session_id,
            )

            feedbacks.append(feedback)
            self._total_feedbacks += 1

        # Persist to database
        if self._conn is not None:
            self._persist_feedbacks(feedbacks)

        logger.debug(
            f"Finalized retrieval {query_id}: {len(feedbacks)} feedbacks, "
            f"{len(outcome.clicked)} clicks"
        )

        return feedbacks

    def compute_relevance(
        self,
        result_id: str,
        position: int,
        clicked: list[str],
        dwell_times: dict[str, float],
    ) -> float:
        """
        Compute relevance score from implicit signals.

        Formula:
            relevance = w_click * click_signal
                      + w_dwell * dwell_signal
                      + w_pos * position_correction

        Where:
            - click_signal = 1 if clicked, 0 otherwise
            - dwell_signal = normalized dwell time with saturation
            - position_correction = bias correction for rank position

        Args:
            result_id: ID of result to compute relevance for
            position: Rank position (0-indexed)
            clicked: List of clicked result IDs
            dwell_times: Mapping of result_id to dwell time

        Returns:
            Relevance score [0, 1]
        """
        # Click signal (binary)
        click_signal = 1.0 if result_id in clicked else 0.0

        # Dwell time signal (normalized with saturation)
        dwell_time = dwell_times.get(result_id, 0.0)
        dwell_signal = self._normalize_dwell_time(dwell_time)

        # Position bias correction
        # Higher positions naturally get more clicks, so we boost lower positions
        # that still got engagement, and penalize high positions that didn't
        position_correction = self._compute_position_correction(
            position=position,
            clicked=click_signal > 0,
            had_dwell=dwell_time >= self.min_dwell_seconds,
        )

        # Combine signals
        relevance = (
            self.click_weight * click_signal +
            self.dwell_weight * dwell_signal +
            self.position_weight * position_correction
        )

        # Clamp to [0, 1]
        return float(np.clip(relevance, 0.0, 1.0))

    def _normalize_dwell_time(self, dwell_time: float) -> float:
        """
        Normalize dwell time to [0, 1] with saturation.

        Uses a log-shaped curve that saturates at optimal_dwell_seconds.
        Very short dwells (< min_dwell) are treated as zero.
        """
        if dwell_time < self.min_dwell_seconds:
            return 0.0

        # Cap at max dwell
        dwell_time = min(dwell_time, self.max_dwell_seconds)

        # Normalize: log-shaped curve to optimal_dwell
        # At optimal_dwell, signal is ~0.9
        # Beyond optimal, diminishing returns
        normalized = 1.0 - np.exp(-dwell_time / self.optimal_dwell_seconds)

        return float(normalized)

    def _compute_position_correction(
        self,
        position: int,
        clicked: bool,
        had_dwell: bool,
    ) -> float:
        """
        Compute position bias correction factor.

        Lower positions (higher ranks) naturally get more attention.
        We correct for this by:
        - Boosting relevance for lower positions that got engagement
        - Penalizing higher positions that were skipped
        """
        # Position decay factor (position 0 = 1.0, position 1 = 0.9, etc.)
        position_bias = self.position_decay ** position

        if clicked or had_dwell:
            # Engagement at lower position = boost (inverse of position bias)
            # This counteracts the natural bias toward top positions
            return (1.0 - position_bias) * 0.5
        else:
            # No engagement: high position = slight penalty, low position = neutral
            return -position_bias * 0.2

    def _compute_confidence(
        self,
        result_id: str,
        clicked: list[str],
        dwell_times: dict[str, float],
    ) -> float:
        """
        Compute confidence in the relevance signal.

        Higher confidence when:
        - Multiple signals agree (click + dwell)
        - Strong signals (long dwell time)

        Lower confidence when:
        - No interaction (just position-based inference)
        - Very short dwell times
        """
        was_clicked = result_id in clicked
        dwell_time = dwell_times.get(result_id, 0.0)
        had_meaningful_dwell = dwell_time >= self.min_dwell_seconds

        if was_clicked and had_meaningful_dwell:
            # Strong signal: click + dwell agree
            confidence = 0.9
            # Boost for longer dwell
            if dwell_time >= self.optimal_dwell_seconds:
                confidence = 0.95
        elif was_clicked:
            # Click but no dwell: medium confidence
            confidence = 0.7
        elif had_meaningful_dwell:
            # Dwell but no click: medium confidence (browsed but didn't select)
            confidence = 0.6
        else:
            # No interaction: low confidence (only position-based inference)
            confidence = 0.3

        return confidence

    def get_training_batch(
        self,
        batch_size: int = 32,
        min_confidence: float = 0.5,
        since: datetime | None = None,
    ) -> list[RetrievalFeedback]:
        """
        Get batch of feedback events for adapter training.

        Args:
            batch_size: Maximum number of feedbacks to return
            min_confidence: Minimum confidence threshold
            since: Only include feedbacks after this timestamp

        Returns:
            List of RetrievalFeedback suitable for training
        """
        batch_size = min(batch_size, MAX_BATCH_SIZE)

        if self._conn is None:
            return []

        query = """
            SELECT * FROM feedback
            WHERE confidence >= ?
        """
        params: list[Any] = [min_confidence]

        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY RANDOM() LIMIT ?"
        params.append(batch_size)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        feedbacks = []
        columns = [desc[0] for desc in cursor.description]

        for row in rows:
            row_dict = dict(zip(columns, row))
            # Convert boolean
            row_dict["clicked"] = bool(row_dict["clicked"])
            feedbacks.append(RetrievalFeedback.from_dict(row_dict))

        return feedbacks

    def get_feedback_for_result(
        self,
        result_id: str,
        limit: int = 100,
    ) -> list[RetrievalFeedback]:
        """
        Get all feedback events for a specific result.

        Useful for analyzing how a particular memory performs.

        Args:
            result_id: Result ID to query
            limit: Maximum number of feedbacks

        Returns:
            List of feedbacks for this result
        """
        if self._conn is None:
            return []

        cursor = self._conn.execute(
            """
            SELECT * FROM feedback
            WHERE result_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (result_id, limit)
        )
        rows = cursor.fetchall()

        feedbacks = []
        columns = [desc[0] for desc in cursor.description]

        for row in rows:
            row_dict = dict(zip(columns, row))
            row_dict["clicked"] = bool(row_dict["clicked"])
            feedbacks.append(RetrievalFeedback.from_dict(row_dict))

        return feedbacks

    def get_average_relevance(self, result_id: str) -> float:
        """
        Get average relevance score for a result across all feedbacks.

        Args:
            result_id: Result ID to query

        Returns:
            Average relevance score, or 0.0 if no feedback
        """
        if self._conn is None:
            return 0.0

        cursor = self._conn.execute(
            """
            SELECT AVG(relevance) FROM feedback
            WHERE result_id = ?
            """,
            (result_id,)
        )
        row = cursor.fetchone()

        if row and row[0] is not None:
            return float(row[0])
        return 0.0

    def _persist_feedbacks(self, feedbacks: list[RetrievalFeedback]) -> None:
        """Persist feedback events to database."""
        if self._conn is None:
            return

        for fb in feedbacks:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO feedback
                (feedback_id, query_id, result_id, relevance, clicked,
                 dwell_time, position, confidence, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(fb.feedback_id),
                    fb.query_id,
                    fb.result_id,
                    fb.relevance,
                    int(fb.clicked),
                    fb.dwell_time,
                    fb.position,
                    fb.confidence,
                    fb.timestamp.isoformat(),
                    fb.session_id,
                )
            )

        self._conn.commit()

    def _evict_oldest_pending(self) -> None:
        """Evict oldest pending retrieval to make room."""
        if not self._pending:
            return

        # Find oldest
        oldest_id = min(
            self._pending.keys(),
            key=lambda k: self._pending[k].timestamp
        )

        # Finalize it (will compute feedback with whatever data exists)
        logger.debug(f"Evicting oldest pending retrieval: {oldest_id}")
        self.finalize_retrieval(oldest_id)

    def cleanup_old_feedback(self) -> int:
        """
        Remove feedback older than retention period.

        Returns:
            Number of records deleted
        """
        if self._conn is None:
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)

        cursor = self._conn.execute(
            """
            DELETE FROM feedback
            WHERE timestamp < ?
            """,
            (cutoff.isoformat(),)
        )

        deleted = cursor.rowcount
        self._conn.commit()

        logger.info(f"Cleaned up {deleted} old feedback records")
        return deleted

    def get_statistics(self) -> dict[str, Any]:
        """Get collector statistics."""
        stats = {
            "total_retrievals": self._total_retrievals,
            "total_feedbacks": self._total_feedbacks,
            "total_clicks": self._total_clicks,
            "pending_retrievals": len(self._pending),
            "click_rate": (
                self._total_clicks / self._total_feedbacks
                if self._total_feedbacks > 0 else 0.0
            ),
        }

        if self._conn is not None:
            cursor = self._conn.execute("SELECT COUNT(*) FROM feedback")
            stats["persisted_feedbacks"] = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT AVG(relevance), AVG(confidence) FROM feedback"
            )
            row = cursor.fetchone()
            stats["avg_relevance"] = row[0] or 0.0
            stats["avg_confidence"] = row[1] or 0.0

        return stats

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


__all__ = [
    "RetrievalFeedback",
    "RetrievalOutcome",
    "RetrievalFeedbackCollector",
]
