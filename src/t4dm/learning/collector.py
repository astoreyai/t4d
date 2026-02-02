"""
Event Collector for T4DM Learning System.

Collects retrieval and outcome events, stores them in SQLite,
and provides matching logic for credit assignment.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from t4dm.learning.events import (
    Experience,
    FeedbackSignal,
    MemoryType,
    OutcomeEvent,
    OutcomeType,
    RetrievalEvent,
)

if TYPE_CHECKING:
    from t4dm.learning.credit_flow import CreditFlowEngine

logger = logging.getLogger(__name__)


# =============================================================================
# SQLite Event Store
# =============================================================================

class EventStore:
    """
    SQLite-backed storage for learning events.

    Provides persistent storage with efficient querying for
    retrieval-outcome matching and experience replay.

    Schema design optimized for:
    1. Fast context_hash lookups (matching retrievals to outcomes)
    2. Time-range queries (for trace decay)
    3. Session-based filtering
    """

    SCHEMA = """
    -- Retrieval events
    CREATE TABLE IF NOT EXISTS retrieval_events (
        id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        retrieved_ids TEXT NOT NULL,  -- JSON array
        retrieval_scores TEXT NOT NULL,  -- JSON dict
        component_scores TEXT,  -- JSON dict of dicts
        context_hash TEXT NOT NULL,
        session_id TEXT,
        project TEXT,
        timestamp TEXT NOT NULL,

        -- Indexing for lookups
        processed INTEGER DEFAULT 0  -- Whether matched to outcome
    );

    CREATE INDEX IF NOT EXISTS idx_retrieval_context ON retrieval_events(context_hash);
    CREATE INDEX IF NOT EXISTS idx_retrieval_session ON retrieval_events(session_id);
    CREATE INDEX IF NOT EXISTS idx_retrieval_time ON retrieval_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_retrieval_processed ON retrieval_events(processed);

    -- Outcome events
    CREATE TABLE IF NOT EXISTS outcome_events (
        id TEXT PRIMARY KEY,
        outcome_type TEXT NOT NULL,
        success_score REAL NOT NULL,
        context_hash TEXT NOT NULL,
        session_id TEXT,
        timestamp TEXT NOT NULL,
        explicit_citations TEXT,  -- JSON array
        feedback_signals TEXT,  -- JSON array
        task_description TEXT,
        tool_results TEXT,  -- JSON dict

        processed INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_outcome_context ON outcome_events(context_hash);
    CREATE INDEX IF NOT EXISTS idx_outcome_session ON outcome_events(session_id);
    CREATE INDEX IF NOT EXISTS idx_outcome_time ON outcome_events(timestamp);

    -- Experiences (retrieval + outcome pairs for training)
    CREATE TABLE IF NOT EXISTS experiences (
        id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        retrieved_ids TEXT NOT NULL,  -- JSON array
        retrieval_scores TEXT NOT NULL,  -- JSON array
        component_vectors TEXT NOT NULL,  -- JSON nested array
        outcome_score REAL NOT NULL,
        per_memory_rewards TEXT NOT NULL,  -- JSON dict
        priority REAL DEFAULT 1.0,
        timestamp TEXT NOT NULL,

        -- Training metadata
        times_sampled INTEGER DEFAULT 0,
        last_sampled TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_exp_priority ON experiences(priority DESC);
    CREATE INDEX IF NOT EXISTS idx_exp_time ON experiences(timestamp);

    -- Eligibility traces (for TD-λ)
    -- Updated to use event-indexed decay per Hinton critique
    CREATE TABLE IF NOT EXISTS eligibility_traces (
        memory_id TEXT PRIMARY KEY,
        trace_value REAL NOT NULL,
        last_retrieval TEXT NOT NULL,
        retrieval_count INTEGER DEFAULT 1,
        last_event_index INTEGER DEFAULT 0  -- Event index when last updated
    );

    -- Global counters for event-indexed operations
    CREATE TABLE IF NOT EXISTS global_counters (
        key TEXT PRIMARY KEY,
        value INTEGER NOT NULL DEFAULT 0
    );

    -- Initialize event counter if not exists
    INSERT OR IGNORE INTO global_counters (key, value) VALUES ('event_index', 0);

    -- Baseline statistics (for advantage computation)
    CREATE TABLE IF NOT EXISTS baseline_stats (
        key TEXT PRIMARY KEY,
        value REAL NOT NULL,
        count INTEGER DEFAULT 1,
        last_updated TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize event store.

        Args:
            db_path: Path to SQLite database. Default: ~/.ww/learning.db
        """
        if db_path is None:
            db_path = str(Path.home() / ".ww" / "learning.db")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_schema()

        logger.info(f"EventStore initialized: {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(self.SCHEMA)
        conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for transactions."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -------------------------------------------------------------------------
    # Retrieval Events
    # -------------------------------------------------------------------------

    def store_retrieval(self, event: RetrievalEvent) -> None:
        """Store a retrieval event."""
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO retrieval_events
                (id, query, memory_type, retrieved_ids, retrieval_scores,
                 component_scores, context_hash, session_id, project, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(event.retrieval_id),
                event.query,
                event.memory_type.value,
                json.dumps([str(uid) for uid in event.retrieved_ids]),
                json.dumps(event.retrieval_scores),
                json.dumps(event.component_scores),
                event.context_hash,
                event.session_id,
                event.project,
                event.timestamp.isoformat()
            ))

        logger.debug(f"Stored retrieval event: {event.retrieval_id}")

    def get_retrievals_by_context(
        self,
        context_hash: str,
        max_age_hours: float = 24.0
    ) -> list[RetrievalEvent]:
        """Get retrieval events matching a context hash."""
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        conn = self._get_conn()
        # LOGIC-008 FIX: Use ASC order so newest retrieval is processed last.
        # When multiple retrievals match same context, all_rewards.update()
        # should give priority to the most recent retrieval (closest to outcome).
        cursor = conn.execute("""
            SELECT * FROM retrieval_events
            WHERE context_hash = ? AND timestamp > ?
            ORDER BY timestamp ASC
        """, (context_hash, cutoff))

        events = []
        for row in cursor.fetchall():
            event = RetrievalEvent(
                retrieval_id=UUID(row["id"]),
                query=row["query"],
                memory_type=MemoryType(row["memory_type"]),
                retrieved_ids=[UUID(uid) for uid in json.loads(row["retrieved_ids"])],
                retrieval_scores=json.loads(row["retrieval_scores"]),
                component_scores=json.loads(row["component_scores"]) if row["component_scores"] else {},
                context_hash=row["context_hash"],
                session_id=row["session_id"] or "",
                project=row["project"] or "",
                timestamp=datetime.fromisoformat(row["timestamp"])
            )
            events.append(event)

        return events

    def get_unprocessed_retrievals(self, limit: int = 100) -> list[RetrievalEvent]:
        """Get retrievals not yet matched to outcomes."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM retrieval_events
            WHERE processed = 0
            ORDER BY timestamp ASC
            LIMIT ?
        """, (limit,))

        events = []
        for row in cursor.fetchall():
            event = RetrievalEvent(
                retrieval_id=UUID(row["id"]),
                query=row["query"],
                memory_type=MemoryType(row["memory_type"]),
                retrieved_ids=[UUID(uid) for uid in json.loads(row["retrieved_ids"])],
                retrieval_scores=json.loads(row["retrieval_scores"]),
                component_scores=json.loads(row["component_scores"]) if row["component_scores"] else {},
                context_hash=row["context_hash"],
                session_id=row["session_id"] or "",
                project=row["project"] or "",
                timestamp=datetime.fromisoformat(row["timestamp"])
            )
            events.append(event)

        return events

    def mark_retrieval_processed(self, retrieval_id: UUID) -> None:
        """Mark a retrieval as processed."""
        with self.transaction() as cursor:
            cursor.execute(
                "UPDATE retrieval_events SET processed = 1 WHERE id = ?",
                (str(retrieval_id),)
            )

    # -------------------------------------------------------------------------
    # Outcome Events
    # -------------------------------------------------------------------------

    def store_outcome(self, event: OutcomeEvent) -> None:
        """Store an outcome event."""
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO outcome_events
                (id, outcome_type, success_score, context_hash, session_id,
                 timestamp, explicit_citations, feedback_signals, task_description, tool_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(event.outcome_id),
                event.outcome_type.value,
                event.success_score,
                event.context_hash,
                event.session_id,
                event.timestamp.isoformat(),
                json.dumps([str(uid) for uid in event.explicit_citations]),
                json.dumps([f.value for f in event.feedback_signals]),
                event.task_description,
                json.dumps(event.tool_results)
            ))

        logger.debug(f"Stored outcome event: {event.outcome_id}")

    def get_outcomes_by_context(
        self,
        context_hash: str,
        max_age_hours: float = 24.0
    ) -> list[OutcomeEvent]:
        """Get outcome events matching a context hash."""
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM outcome_events
            WHERE context_hash = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (context_hash, cutoff))

        events = []
        for row in cursor.fetchall():
            event = OutcomeEvent(
                outcome_id=UUID(row["id"]),
                outcome_type=OutcomeType(row["outcome_type"]),
                success_score=row["success_score"],
                context_hash=row["context_hash"],
                session_id=row["session_id"] or "",
                timestamp=datetime.fromisoformat(row["timestamp"]),
                explicit_citations=[UUID(uid) for uid in json.loads(row["explicit_citations"])] if row["explicit_citations"] else [],
                feedback_signals=[FeedbackSignal(f) for f in json.loads(row["feedback_signals"])] if row["feedback_signals"] else [],
                task_description=row["task_description"] or "",
                tool_results=json.loads(row["tool_results"]) if row["tool_results"] else {}
            )
            events.append(event)

        return events

    # -------------------------------------------------------------------------
    # Experiences
    # -------------------------------------------------------------------------

    def store_experience(self, exp: Experience) -> None:
        """Store a training experience."""
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO experiences
                (id, query, memory_type, retrieved_ids, retrieval_scores,
                 component_vectors, outcome_score, per_memory_rewards, priority, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(exp.experience_id),
                exp.query,
                exp.memory_type.value,
                json.dumps([str(uid) for uid in exp.retrieved_ids]),
                json.dumps(exp.retrieval_scores),
                json.dumps(exp.component_vectors),
                exp.outcome_score,
                json.dumps(exp.per_memory_rewards),
                exp.priority,
                exp.timestamp.isoformat()
            ))

        logger.debug(f"Stored experience: {exp.experience_id}")

    def sample_experiences(
        self,
        n: int,
        prioritized: bool = True
    ) -> list[Experience]:
        """
        Sample experiences for training.

        Args:
            n: Number of experiences to sample
            prioritized: If True, sample proportional to priority

        Returns:
            List of experiences
        """
        conn = self._get_conn()

        if prioritized:
            # Weighted sampling by priority
            cursor = conn.execute("""
                SELECT *, priority / (SELECT SUM(priority) FROM experiences) as prob
                FROM experiences
                ORDER BY RANDOM() * (1.0 / (priority + 0.01))
                LIMIT ?
            """, (n,))
        else:
            cursor = conn.execute("""
                SELECT * FROM experiences
                ORDER BY RANDOM()
                LIMIT ?
            """, (n,))

        experiences = []
        for row in cursor.fetchall():
            exp = Experience(
                experience_id=UUID(row["id"]),
                query=row["query"],
                memory_type=MemoryType(row["memory_type"]),
                retrieved_ids=[UUID(uid) for uid in json.loads(row["retrieved_ids"])],
                retrieval_scores=json.loads(row["retrieval_scores"]),
                component_vectors=json.loads(row["component_vectors"]),
                outcome_score=row["outcome_score"],
                per_memory_rewards=json.loads(row["per_memory_rewards"]),
                priority=row["priority"],
                timestamp=datetime.fromisoformat(row["timestamp"])
            )
            experiences.append(exp)

            # Update sample count
            with self.transaction() as cur:
                cur.execute("""
                    UPDATE experiences
                    SET times_sampled = times_sampled + 1, last_sampled = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), str(exp.experience_id)))

        return experiences

    def update_priority(self, experience_id: UUID, new_priority: float) -> None:
        """Update experience priority (e.g., based on TD error)."""
        with self.transaction() as cursor:
            cursor.execute(
                "UPDATE experiences SET priority = ? WHERE id = ?",
                (new_priority, str(experience_id))
            )

    def count_experiences(self) -> int:
        """Count total experiences in store."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM experiences")
        return cursor.fetchone()[0]

    # -------------------------------------------------------------------------
    # Eligibility Traces
    # -------------------------------------------------------------------------

    def get_event_index(self) -> int:
        """Get current global event index."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT value FROM global_counters WHERE key = 'event_index'"
        )
        row = cursor.fetchone()
        return row["value"] if row else 0

    def increment_event_index(self) -> int:
        """
        Increment and return the global event index.

        Call this after each learning event (retrieval + outcome pair).
        """
        with self.transaction() as cur:
            cur.execute("""
                UPDATE global_counters
                SET value = value + 1
                WHERE key = 'event_index'
            """)

        return self.get_event_index()

    def update_trace(self, memory_id: UUID, delta: float) -> float:
        """
        Update eligibility trace for a memory.

        Args:
            memory_id: Memory ID
            delta: Amount to add to trace

        Returns:
            New trace value
        """
        mem_id = str(memory_id)
        now = datetime.now().isoformat()
        event_index = self.get_event_index()

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT trace_value, retrieval_count FROM eligibility_traces WHERE memory_id = ?",
            (mem_id,)
        )
        row = cursor.fetchone()

        if row:
            new_trace = row["trace_value"] + delta
            new_count = row["retrieval_count"] + 1
            with self.transaction() as cur:
                cur.execute("""
                    UPDATE eligibility_traces
                    SET trace_value = ?, last_retrieval = ?, retrieval_count = ?,
                        last_event_index = ?
                    WHERE memory_id = ?
                """, (new_trace, now, new_count, event_index, mem_id))
        else:
            new_trace = delta
            with self.transaction() as cur:
                cur.execute("""
                    INSERT INTO eligibility_traces
                        (memory_id, trace_value, last_retrieval, last_event_index)
                    VALUES (?, ?, ?, ?)
                """, (mem_id, new_trace, now, event_index))

        return new_trace

    def decay_traces(
        self,
        gamma: float = 0.99,
        lambda_: float = 0.9,
        use_event_indexed: bool = True
    ) -> int:
        """
        Apply TD-lambda decay to all traces.

        Per Hinton critique: Traces should decay per learning event, not
        wall-clock time. Event-indexed decay ensures consistent credit
        assignment regardless of time between sessions.

        Args:
            gamma: Discount factor
            lambda_: Trace decay rate
            use_event_indexed: If True (default), decay based on events elapsed.
                               If False, use legacy wall-clock decay.

        Returns:
            Number of traces updated
        """
        conn = self._get_conn()
        current_event = self.get_event_index()
        now = datetime.now()

        cursor = conn.execute("SELECT * FROM eligibility_traces")
        count = 0

        for row in cursor.fetchall():
            if use_event_indexed:
                # Event-indexed decay (recommended)
                # sqlite3.Row requires subscript access
                try:
                    last_event = row["last_event_index"] or 0
                except (KeyError, IndexError):
                    last_event = 0
                events_elapsed = current_event - last_event

                # Decay based on number of events since last retrieval
                # Each event applies one step of gamma*lambda decay
                decay_factor = (gamma * lambda_) ** events_elapsed
            else:
                # Legacy wall-clock decay (kept for backward compatibility)
                last_retrieval = datetime.fromisoformat(row["last_retrieval"])
                hours_elapsed = (now - last_retrieval).total_seconds() / 3600
                decay_factor = (gamma * lambda_) ** hours_elapsed

            new_trace = row["trace_value"] * decay_factor

            # Prune negligible traces
            if new_trace < 0.001:
                with self.transaction() as cur:
                    cur.execute(
                        "DELETE FROM eligibility_traces WHERE memory_id = ?",
                        (row["memory_id"],)
                    )
            else:
                with self.transaction() as cur:
                    cur.execute(
                        "UPDATE eligibility_traces SET trace_value = ? WHERE memory_id = ?",
                        (new_trace, row["memory_id"])
                    )
            count += 1

        return count

    def decay_traces_for_event(self, gamma: float = 0.99, lambda_: float = 0.9) -> int:
        """
        Apply single-step TD-lambda decay and increment event counter.

        This is the recommended way to apply decay: call once per learning event
        (retrieval + outcome pair). Each call applies exactly one step of gamma*lambda decay
        to all active traces.

        Args:
            gamma: Discount factor
            lambda_: Trace decay rate

        Returns:
            Number of traces updated
        """
        # First, apply one step of decay to all traces
        conn = self._get_conn()
        decay_factor = gamma * lambda_

        with self.transaction() as cur:
            # Update all traces with single-step decay
            cur.execute("""
                UPDATE eligibility_traces
                SET trace_value = trace_value * ?
            """, (decay_factor,))

            # Delete negligible traces
            cur.execute("""
                DELETE FROM eligibility_traces
                WHERE trace_value < 0.001
            """)

        # Then increment the event counter
        self.increment_event_index()

        # Return count of remaining traces
        cursor = conn.execute("SELECT COUNT(*) FROM eligibility_traces")
        return cursor.fetchone()[0]

    def get_traces(self) -> dict[str, float]:
        """Get all current eligibility traces."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT memory_id, trace_value FROM eligibility_traces")
        return {row["memory_id"]: row["trace_value"] for row in cursor.fetchall()}

    # -------------------------------------------------------------------------
    # Baseline Statistics
    # -------------------------------------------------------------------------

    def update_baseline(self, key: str, value: float) -> float:
        """
        Update running baseline statistic.

        Uses exponential moving average for stability.

        Args:
            key: Statistic key (e.g., "global", "episodic", "project:ww")
            value: New value to incorporate

        Returns:
            Updated baseline value
        """
        alpha = 0.1  # EMA smoothing factor
        now = datetime.now().isoformat()

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT value, count FROM baseline_stats WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()

        if row:
            # EMA update
            new_value = alpha * value + (1 - alpha) * row["value"]
            new_count = row["count"] + 1
            with self.transaction() as cur:
                cur.execute("""
                    UPDATE baseline_stats
                    SET value = ?, count = ?, last_updated = ?
                    WHERE key = ?
                """, (new_value, new_count, now, key))
        else:
            new_value = value
            with self.transaction() as cur:
                cur.execute("""
                    INSERT INTO baseline_stats (key, value, count, last_updated)
                    VALUES (?, ?, 1, ?)
                """, (key, value, now))

        return new_value

    def get_baseline(self, key: str, default: float = 0.5) -> float:
        """Get baseline statistic."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT value FROM baseline_stats WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        return row["value"] if row else default

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        conn = self._get_conn()

        stats = {}
        for table in ["retrieval_events", "outcome_events", "experiences", "eligibility_traces"]:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        return stats

    def cleanup_old_events(self, max_age_days: int = 30) -> int:
        """Remove events older than max_age_days."""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        count = 0
        with self.transaction() as cursor:
            cursor.execute(
                "DELETE FROM retrieval_events WHERE timestamp < ? AND processed = 1",
                (cutoff,)
            )
            count += cursor.rowcount

            cursor.execute(
                "DELETE FROM outcome_events WHERE timestamp < ? AND processed = 1",
                (cutoff,)
            )
            count += cursor.rowcount

        return count


# =============================================================================
# Event Collector
# =============================================================================

@dataclass
class CollectorConfig:
    """Configuration for event collector."""
    db_path: str | None = None
    auto_match: bool = True  # Automatically match retrievals to outcomes
    match_window_hours: float = 24.0
    trace_gamma: float = 0.99
    trace_lambda: float = 0.9


@dataclass
class RetrievalContext:
    """Context from a retrieval for credit flow integration."""
    retrieved_ids: list[UUID]
    embeddings: dict[str, np.ndarray]  # memory_id -> embedding
    query_embedding: np.ndarray | None = None


class EventCollector:
    """
    Collects learning events and coordinates storage.

    Acts as the main interface for the learning system,
    collecting events from memory operations and coordinating
    with the event store.

    Credit Flow Integration:
    When credit_flow_engine is provided, outcomes automatically trigger
    end-to-end credit assignment from neuromodulator signals to memory updates.
    """

    def __init__(
        self,
        config: CollectorConfig | None = None,
        credit_flow_engine: CreditFlowEngine | None = None
    ):
        """
        Initialize collector.

        Args:
            config: Collector configuration
            credit_flow_engine: Optional CreditFlowEngine for automatic
                credit assignment. When provided, record_outcome() will
                automatically apply learning signals to memory embeddings.
        """
        self.config = config or CollectorConfig()
        self.store = EventStore(self.config.db_path)
        self._pending_context: str | None = None
        self.credit_flow_engine = credit_flow_engine

        # Store retrieval context for credit flow integration
        self._pending_retrieval_context: RetrievalContext | None = None

        logger.info("EventCollector initialized")

    def record_retrieval(
        self,
        query: str,
        memory_type: MemoryType,
        retrieved_ids: list[UUID],
        retrieval_scores: dict[str, float],
        component_scores: dict[str, dict[str, float]] | None = None,
        context: str | None = None,
        session_id: str = "",
        project: str = "",
        embeddings: dict[str, np.ndarray] | None = None,
        query_embedding: np.ndarray | None = None
    ) -> RetrievalEvent:
        """
        Record a retrieval event.

        Args:
            query: The query that triggered retrieval
            memory_type: Type of memory system queried
            retrieved_ids: IDs of retrieved memories (ordered by score)
            retrieval_scores: Final scores for each memory
            component_scores: Breakdown of score components
            context: Conversation context for hashing
            session_id: Current session ID
            project: Current project
            embeddings: Optional memory embeddings for credit flow integration
            query_embedding: Optional query embedding for credit flow integration

        Returns:
            The created RetrievalEvent
        """
        event = RetrievalEvent(
            query=query,
            memory_type=memory_type,
            retrieved_ids=retrieved_ids,
            retrieval_scores=retrieval_scores,
            component_scores=component_scores or {},
            session_id=session_id,
            project=project
        )

        if context:
            event.compute_context_hash(context)
            self._pending_context = event.context_hash

        # Store event
        self.store.store_retrieval(event)

        # Update eligibility traces - use full UUIDs from retrieved_ids
        # retrieval_scores uses short IDs (8 chars), so map back to full UUIDs
        short_to_full = {str(uid)[:8]: uid for uid in retrieved_ids}
        for short_id, score in retrieval_scores.items():
            if short_id in short_to_full:
                self.store.update_trace(short_to_full[short_id], score)

        # Store retrieval context for credit flow integration
        if embeddings is not None or query_embedding is not None:
            self._pending_retrieval_context = RetrievalContext(
                retrieved_ids=retrieved_ids,
                embeddings=embeddings or {},
                query_embedding=query_embedding
            )
        else:
            self._pending_retrieval_context = None

        logger.debug(f"Recorded retrieval: {len(retrieved_ids)} memories, context={event.context_hash[:8]}")
        return event

    def record_outcome(
        self,
        outcome_type: OutcomeType,
        success_score: float,
        context: str | None = None,
        context_hash: str | None = None,
        session_id: str = "",
        explicit_citations: list[UUID] | None = None,
        feedback_signals: list[FeedbackSignal] | None = None,
        task_description: str = "",
        tool_results: dict[str, Any] | None = None
    ) -> OutcomeEvent:
        """
        Record an outcome event.

        When credit_flow_engine is configured, this automatically triggers
        end-to-end credit assignment:
        1. Computes per-memory rewards from outcome
        2. Applies neuromodulator learning signals
        3. Updates memory embeddings via reconsolidation

        Args:
            outcome_type: Classification of outcome
            success_score: Continuous success measure [0, 1]
            context: Conversation context (for hash matching)
            context_hash: Pre-computed context hash
            session_id: Current session ID
            explicit_citations: Memories explicitly cited as helpful
            feedback_signals: Detected feedback signals
            task_description: Description of the task
            tool_results: Results from tool executions

        Returns:
            The created OutcomeEvent
        """
        event = OutcomeEvent(
            outcome_type=outcome_type,
            success_score=success_score,
            session_id=session_id,
            explicit_citations=explicit_citations or [],
            feedback_signals=feedback_signals or [],
            task_description=task_description,
            tool_results=tool_results or {}
        )

        # Determine context hash
        if context_hash:
            event.context_hash = context_hash
        elif context:
            event.context_hash = event.compute_context_hash(context)
        elif self._pending_context:
            event.context_hash = self._pending_context

        # Store event
        self.store.store_outcome(event)

        # Auto-match BEFORE updating baseline (so we compute advantage correctly)
        computed_rewards = {}
        if self.config.auto_match and event.context_hash:
            computed_rewards = self._match_and_create_experiences(event)

        # Update baseline AFTER matching
        self.store.update_baseline("global", success_score)

        # Credit Flow Integration: Apply learning signals to memory embeddings
        if (
            self.credit_flow_engine is not None
            and self._pending_retrieval_context is not None
            and computed_rewards
        ):
            self._apply_credit_flow(computed_rewards, event)

        logger.debug(f"Recorded outcome: {outcome_type.value}, score={success_score:.2f}")
        return event

    def _apply_credit_flow(
        self,
        computed_rewards: dict[str, float],
        outcome_event: OutcomeEvent
    ) -> None:
        """
        Apply credit flow to update memory embeddings.

        This bridges the gap between outcome recording and actual memory updates.

        Args:
            computed_rewards: Per-memory rewards from credit assignment
            outcome_event: The outcome event
        """
        if self.credit_flow_engine is None or self._pending_retrieval_context is None:
            return

        ctx = self._pending_retrieval_context

        # Build list of (memory_id, embedding) tuples for memories we have embeddings for
        retrieved_memories = []
        for memory_id in ctx.retrieved_ids:
            mem_id_str = str(memory_id)
            if mem_id_str in ctx.embeddings:
                retrieved_memories.append((memory_id, ctx.embeddings[mem_id_str]))

        if not retrieved_memories:
            logger.debug("No embeddings available for credit flow")
            return

        if ctx.query_embedding is None:
            logger.debug("No query embedding available for credit flow")
            return

        try:
            # Process and apply credit flow
            result = self.credit_flow_engine.process_and_apply_outcome(
                memory_outcomes=computed_rewards,
                retrieved_memories=retrieved_memories,
                query_embedding=ctx.query_embedding,
                session_outcome=outcome_event.success_score
            )

            episodic_updates = result.get("episodic_updates", {})
            if episodic_updates:
                logger.info(
                    f"Credit flow applied: {len(episodic_updates)} episodic memory updates"
                )

        except Exception as e:
            logger.warning(f"Credit flow application failed: {e}")

        # Clear pending context after use
        self._pending_retrieval_context = None

    def _match_and_create_experiences(self, outcome: OutcomeEvent) -> dict[str, float]:
        """
        Match outcome to retrievals and create experiences.

        Returns:
            Dict of memory_id -> reward for credit flow integration
        """
        retrievals = self.store.get_retrievals_by_context(
            outcome.context_hash,
            max_age_hours=self.config.match_window_hours
        )

        if not retrievals:
            return {}

        # Get baseline for advantage computation
        baseline = self.store.get_baseline("global")

        all_rewards = {}
        count = 0
        for retrieval in retrievals:
            # Compute per-memory rewards
            rewards = self._compute_rewards(retrieval, outcome, baseline)
            all_rewards.update(rewards)

            # Create experience
            exp = Experience(
                query=retrieval.query,
                memory_type=retrieval.memory_type,
                retrieved_ids=retrieval.retrieved_ids,
                retrieval_scores=list(retrieval.retrieval_scores.values()),
                component_vectors=self._extract_component_vectors(retrieval),
                outcome_score=outcome.success_score,
                per_memory_rewards=rewards
            )

            self.store.store_experience(exp)
            self.store.mark_retrieval_processed(retrieval.retrieval_id)
            count += 1

        return all_rewards

    def _compute_rewards(
        self,
        retrieval: RetrievalEvent,
        outcome: OutcomeEvent,
        baseline: float
    ) -> dict[str, float]:
        """
        Compute per-memory rewards using credit assignment.

        Implements:
        R(memory) = (outcome - baseline) * time_discount * attention_weight * citation_bonus

        Args:
            retrieval: The retrieval event
            outcome: The outcome event
            baseline: Current baseline for advantage

        Returns:
            Dict mapping memory_id -> reward
        """
        # Advantage (outcome relative to baseline)
        advantage = outcome.success_score - baseline

        # Time discount (hyperbolic)
        hours_delay = (outcome.timestamp - retrieval.timestamp).total_seconds() / 3600
        time_discount = 1.0 / (1.0 + 0.1 * hours_delay)

        # Compute attention weights (softmax over scores)
        scores = list(retrieval.retrieval_scores.values())
        if not scores:
            return {}

        max_score = max(scores)
        exp_scores = [2.718 ** (s - max_score) for s in scores]  # Stable softmax
        sum_exp = sum(exp_scores)
        attention_weights = [e / sum_exp for e in exp_scores]

        # Citation bonus
        cited_ids = {str(uid) for uid in outcome.explicit_citations}

        rewards = {}
        for (mem_id, score), attn in zip(retrieval.retrieval_scores.items(), attention_weights):
            citation_bonus = 1.5 if mem_id in cited_ids else 1.0
            reward = advantage * time_discount * attn * citation_bonus
            rewards[mem_id] = reward

        return rewards

    def _extract_component_vectors(self, retrieval: RetrievalEvent) -> list[list[float]]:
        """Extract component score vectors for each memory."""
        vectors = []
        for mem_id in retrieval.retrieved_ids:
            components = retrieval.component_scores.get(str(mem_id), {})
            # Standard component order: [similarity, recency, importance, ...]
            vec = [
                components.get("similarity", 0.0),
                components.get("recency", 0.0),
                components.get("importance", 0.0),
                components.get("outcome_history", 0.0),
            ]
            vectors.append(vec)
        return vectors

    def decay_traces(self) -> int:
        """Apply trace decay."""
        return self.store.decay_traces(
            gamma=self.config.trace_gamma,
            lambda_=self.config.trace_lambda
        )

    def get_stats(self) -> dict[str, Any]:
        """Get collector statistics."""
        store_stats = self.store.get_stats()
        store_stats["baseline_global"] = self.store.get_baseline("global")
        store_stats["credit_flow_enabled"] = self.credit_flow_engine is not None
        return store_stats


# =============================================================================
# Singleton Instance
# =============================================================================

_collector_instance: EventCollector | None = None


def get_collector(
    config: CollectorConfig | None = None,
    credit_flow_engine: CreditFlowEngine | None = None
) -> EventCollector:
    """Get or create singleton collector."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = EventCollector(config, credit_flow_engine)
    return _collector_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CollectorConfig",
    "EventCollector",
    "EventStore",
    "RetrievalContext",
    "get_collector",
]
