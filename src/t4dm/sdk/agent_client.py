"""
Agent Memory Client for Claude Agent SDK Integration.

Provides outcome-based learning through the three-factor rule:
- Eligibility traces track which memories were retrieved
- Dopamine signals from task success/failure
- Neuromodulator gating from context (encoding vs retrieval)

Based on Hinton agent recommendations for closing the credit assignment loop.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import numpy as np

from t4dm.bridges.ff_retrieval_scorer import FFRetrievalScorer, FFRetrievalConfig
from t4dm.learning.dopamine import DopamineSystem, RewardPredictionError
from t4dm.learning.eligibility import EligibilityTrace
from t4dm.learning.three_factor import ThreeFactorLearningRule, ThreeFactorSignal
from t4dm.learning.neuromodulators import NeuromodulatorState
from t4dm.sdk.client import AsyncWorldWeaverClient, WorldWeaverError
from t4dm.sdk.models import Episode, RecallResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Context for a retrieval event, used for credit assignment."""

    task_id: str
    query: str
    memory_ids: list[str]
    embeddings: list[list[float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    scores: list[float] = field(default_factory=list)


@dataclass
class CreditAssignmentResult:
    """Result of credit assignment after task outcome."""

    credited: int
    signals: dict[str, ThreeFactorSignal] = field(default_factory=dict)
    reconsolidated: list[str] = field(default_factory=list)
    total_lr_applied: float = 0.0


@dataclass
class ScoredMemory:
    """Memory with relevance score from FF network."""

    episode: Episode
    similarity_score: float
    ff_relevance_score: float = 0.0
    combined_score: float = 0.0

    def __post_init__(self):
        # Combined score weights similarity and learned relevance
        self.combined_score = 0.6 * self.similarity_score + 0.4 * self.ff_relevance_score


class AgentMemoryClient:
    """
    Memory client for Claude Agent SDK with outcome-based learning.

    Key insight from Hinton agent: Memory retrieval is an action, and actions
    that lead to good outcomes should be reinforced. The representations of
    useful memories should drift toward the queries that activate them successfully.

    This client:
    1. Tracks which memories were retrieved for each task
    2. Collects task outcomes (success/failure/partial)
    3. Propagates credit via three-factor learning rule
    4. Triggers reconsolidation for active memories

    Example:
        async with AgentMemoryClient(session_id="agent-123") as mem:
            # Retrieve memories for a task
            memories = await mem.retrieve_for_task(
                task_id="fix-auth-bug",
                query="authentication error handling patterns"
            )

            # ... agent uses memories to complete task ...

            # Report outcome
            result = await mem.report_task_outcome(
                task_id="fix-auth-bug",
                success=True
            )
            # Memories that helped are now reinforced!
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        session_id: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        base_learning_rate: float = 0.01,
        eligibility_decay: float = 0.95,
        min_eligibility_threshold: float = 0.1,
        ff_scorer: FFRetrievalScorer | None = None,
    ):
        """
        Initialize agent memory client.

        Args:
            base_url: T4DM API URL
            session_id: Session ID for memory isolation
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            base_learning_rate: Base LR for three-factor updates
            eligibility_decay: Decay rate for eligibility traces
            min_eligibility_threshold: Minimum eligibility for credit assignment
            ff_scorer: Optional FFRetrievalScorer for local FF-based relevance.
                       If None, uses text heuristic. For full FF scoring with
                       embeddings, this happens server-side in the recall pipeline.
        """
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or f"agent-{uuid4().hex[:8]}"
        self.api_key = api_key
        self.timeout = timeout
        self.base_learning_rate = base_learning_rate
        self.min_eligibility_threshold = min_eligibility_threshold

        # Core client
        self._client: AsyncWorldWeaverClient | None = None

        # FF-based retrieval scoring (optional - full scoring done server-side)
        self._ff_scorer = ff_scorer

        # Learning systems (simplified for SDK - full three-factor runs server-side)
        self._eligibility = EligibilityTrace(decay=eligibility_decay)
        self._dopamine = DopamineSystem()
        self._three_factor = ThreeFactorLearningRule(
            dopamine_system=self._dopamine,
        )
        # Create default neuromodulator state for encoding mode
        self._neuromod_state = self._create_encoding_state()

        # Active retrievals for credit assignment
        self._active_retrievals: dict[str, RetrievalContext] = {}

        # Statistics
        self._total_retrievals = 0
        self._total_outcomes = 0
        self._total_credits_assigned = 0
        self._successful_tasks = 0

    async def __aenter__(self) -> "AgentMemoryClient":
        """Enter async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        await self.close()

    async def connect(self):
        """Initialize connection to T4DM API."""
        headers = {"X-Session-ID": self.session_id}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self._client = AsyncWorldWeaverClient(
            base_url=self.base_url,
            session_id=self.session_id,
            timeout=self.timeout,
        )
        await self._client.connect()

        # Set neuromodulator state for encoding
        self._neuromod_state = self._create_encoding_state()

        logger.info(f"AgentMemoryClient connected: session={self.session_id}")

    async def close(self):
        """Close connection and persist state."""
        if self._client:
            # Assign credit for any pending retrievals
            for task_id in list(self._active_retrievals.keys()):
                logger.warning(f"Task {task_id} never reported outcome, assigning neutral credit")
                await self.report_task_outcome(task_id, success=None, partial_credit=0.5)

            await self._client.close()
            self._client = None

        logger.info(
            f"AgentMemoryClient closed: retrievals={self._total_retrievals}, "
            f"outcomes={self._total_outcomes}, credits={self._total_credits_assigned}"
        )

    def _get_client(self) -> AsyncWorldWeaverClient:
        """Get client, raising if not connected."""
        if not self._client:
            raise WorldWeaverError("Client not connected. Use 'async with' or call connect()")
        return self._client

    def _create_encoding_state(self) -> NeuromodulatorState:
        """Create neuromodulator state for encoding mode."""
        return NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )

    def _create_retrieval_state(self) -> NeuromodulatorState:
        """Create neuromodulator state for retrieval mode."""
        return NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )

    # =========================================================================
    # Memory Retrieval with Eligibility Tracking
    # =========================================================================

    async def retrieve_for_task(
        self,
        task_id: str,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.5,
        project: str | None = None,
    ) -> list[ScoredMemory]:
        """
        Retrieve memories for a task and mark them as active in eligibility traces.

        The task_id links this retrieval to eventual outcome for credit assignment.

        Args:
            task_id: Unique identifier for the agent task
            query: Semantic search query
            limit: Maximum memories to retrieve
            min_similarity: Minimum cosine similarity threshold
            project: Optional project filter

        Returns:
            List of scored memories with relevance information
        """
        client = self._get_client()

        # Perform retrieval
        result = await client.recall_episodes(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            project=project,
        )

        # Mark all retrieved memories as active in eligibility traces
        memory_ids = []
        for episode in result.episodes:
            memory_id = str(episode.id)
            memory_ids.append(memory_id)
            self._eligibility.update(memory_id)

        # Store retrieval context for credit assignment
        self._active_retrievals[task_id] = RetrievalContext(
            task_id=task_id,
            query=query,
            memory_ids=memory_ids,
            scores=result.scores,
            timestamp=datetime.now(),
        )

        self._total_retrievals += 1

        # Convert to scored memories
        scored = []
        for episode, score in zip(result.episodes, result.scores):
            scored.append(ScoredMemory(
                episode=episode,
                similarity_score=score,
                ff_relevance_score=self._compute_ff_relevance(query, episode),
            ))

        # Sort by combined score
        scored.sort(key=lambda x: x.combined_score, reverse=True)

        logger.debug(
            f"Retrieved {len(scored)} memories for task={task_id}, "
            f"query='{query[:50]}...'"
        )

        return scored

    def _compute_ff_relevance(
        self,
        query: str,
        episode: Episode,
        query_embedding: np.ndarray | None = None,
        episode_embedding: np.ndarray | None = None,
    ) -> float:
        """
        Compute FF-based relevance score for query-memory pair.

        Architecture Note:
        - Full FF scoring with embeddings happens SERVER-SIDE in the recall pipeline
          where embeddings are directly available from Qdrant.
        - This client-side method provides supplementary scoring when:
          a) An FFRetrievalScorer is injected with local embeddings
          b) Falls back to text-based heuristic otherwise

        The text heuristic provides a reasonable baseline that captures semantic
        overlap without requiring embedding computation on the client.

        Args:
            query: Search query text
            episode: Episode to score
            query_embedding: Optional embedding vector for query
            episode_embedding: Optional embedding vector for episode

        Returns:
            Relevance score in [0, 1] range
        """
        # If we have an FF scorer AND embeddings, use proper FF scoring
        if (
            self._ff_scorer is not None
            and query_embedding is not None
            and episode_embedding is not None
        ):
            try:
                # Combine query and episode embeddings for FF goodness computation
                # The scorer expects concatenated or combined representations
                combined = np.concatenate([query_embedding, episode_embedding])
                scores = self._ff_scorer.score_candidates(
                    query_embedding=query_embedding,
                    candidate_embeddings=[episode_embedding],
                    candidate_ids=[str(episode.id)],
                )
                if scores:
                    return scores[0].confidence
            except Exception as e:
                logger.debug(f"FF scoring failed, using heuristic: {e}")

        # Text-based heuristic fallback
        # This captures semantic overlap without requiring embeddings
        query_words = set(query.lower().split())
        content_words = set(episode.content.lower().split())

        if not query_words:
            return 0.5

        # Jaccard-like overlap with boost for exact matches
        overlap = len(query_words & content_words) / len(query_words)
        return min(1.0, overlap * 1.2)

    # =========================================================================
    # Outcome Reporting and Credit Assignment
    # =========================================================================

    async def report_task_outcome(
        self,
        task_id: str,
        success: bool | None = None,
        partial_credit: float | None = None,
        feedback: str | None = None,
    ) -> CreditAssignmentResult:
        """
        Report task outcome and propagate credit to retrieved memories.

        This is WHERE THE LEARNING HAPPENS:
        1. Get eligibility traces for memories retrieved for this task
        2. Compute RPE (actual outcome vs expected)
        3. Apply three-factor learning rate
        4. Update memory embeddings via reconsolidation

        Args:
            task_id: Task identifier from retrieve_for_task()
            success: True for success, False for failure, None for neutral
            partial_credit: Float 0-1 for partial success (overrides success)
            feedback: Optional human feedback for logging

        Returns:
            CreditAssignmentResult with statistics about what was updated
        """
        context = self._active_retrievals.pop(task_id, None)
        if not context:
            logger.warning(f"No retrieval context found for task={task_id}")
            return CreditAssignmentResult(credited=0)

        # Determine outcome score
        if partial_credit is not None:
            outcome_score = max(0.0, min(1.0, partial_credit))
        elif success is True:
            outcome_score = 1.0
        elif success is False:
            outcome_score = 0.0
        else:
            outcome_score = 0.5  # Neutral

        # Compute simplified dopamine-like reward signal
        # The full three-factor learning happens server-side
        rpe_value = outcome_score - 0.5  # Simple RPE: deviation from neutral

        # Compute three-factor signals for each retrieved memory
        signals: dict[str, ThreeFactorSignal] = {}
        for memory_id in context.memory_ids:
            eligibility = self._eligibility.get_trace(memory_id)

            if eligibility < self.min_eligibility_threshold:
                continue  # Too decayed to credit

            # Compute three-factor learning signal
            try:
                signal = self._three_factor.compute(
                    memory_id=UUID(memory_id),
                    base_lr=self.base_learning_rate,
                    outcome=outcome_score,
                    neuromod_state=self._neuromod_state,
                )
                signals[memory_id] = signal
            except Exception as e:
                # Fallback: create simplified signal
                logger.debug(f"Three-factor compute failed: {e}, using fallback")
                effective_lr = self.base_learning_rate * eligibility * rpe_value
                signals[memory_id] = ThreeFactorSignal(
                    eligibility=eligibility,
                    neuromod_gate=1.0,
                    dopamine_surprise=rpe_value,
                    effective_lr=effective_lr,
                )

        # Apply reconsolidation for memories with significant credit
        reconsolidated = []
        total_lr = 0.0

        for memory_id, signal in signals.items():
            if abs(signal.effective_lr) > 0.001:  # Meaningful update
                try:
                    await self._reconsolidate_memory(
                        memory_id=memory_id,
                        outcome=outcome_score,
                        effective_lr=signal.effective_lr,
                        rpe_value=rpe_value,
                    )
                    reconsolidated.append(memory_id)
                    total_lr += abs(signal.effective_lr)
                except Exception as e:
                    logger.error(f"Reconsolidation failed for {memory_id}: {e}")

        # Update statistics
        self._total_outcomes += 1
        self._total_credits_assigned += len(reconsolidated)
        if success:
            self._successful_tasks += 1

        # Decay eligibility traces
        self._eligibility.step()

        result = CreditAssignmentResult(
            credited=len(reconsolidated),
            signals=signals,
            reconsolidated=reconsolidated,
            total_lr_applied=total_lr,
        )

        logger.info(
            f"Credit assigned for task={task_id}: "
            f"outcome={outcome_score:.2f}, credited={result.credited}, "
            f"rpe={rpe_value:.3f}"
        )

        return result

    async def _reconsolidate_memory(
        self,
        memory_id: str,
        outcome: float,
        effective_lr: float,
        rpe_value: float,
    ):
        """
        Apply reconsolidation update to a memory.

        This updates the memory's embedding based on the outcome signal.
        Positive outcomes shift embedding toward the query that activated it.
        Negative outcomes do the opposite (or maintain stability).
        """
        client = self._get_client()

        # Call reconsolidation API endpoint
        # This would trigger the three-factor learning in the backend
        try:
            await client._request(
                "POST",
                f"/episodes/{memory_id}/reconsolidate",
                json={
                    "outcome": outcome,
                    "effective_lr": effective_lr,
                    "dopamine_signal": rpe_value,
                    "source": "agent_outcome",
                },
            )
        except Exception as e:
            # Reconsolidation endpoint may not exist yet
            logger.debug(f"Reconsolidation API not available: {e}")

    # =========================================================================
    # Memory Storage with Encoding
    # =========================================================================

    async def store_experience(
        self,
        content: str,
        outcome: str = "neutral",
        importance: float = 0.5,
        project: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """
        Store a new experience in episodic memory.

        Uses encoding-mode neuromodulator state for proper memory formation.

        Args:
            content: Experience content
            outcome: Outcome category (success/failure/neutral)
            importance: Importance score 0-1 (affects initial stability)
            project: Optional project context
            metadata: Additional metadata

        Returns:
            Created episode
        """
        client = self._get_client()

        # Ensure encoding mode
        self._neuromod_state = self._create_encoding_state()

        # Map importance to emotional valence (affects memory strength)
        emotional_valence = 0.5 + (importance - 0.5) * 0.4

        episode = await client.create_episode(
            content=content,
            project=project,
            outcome=outcome,
            emotional_valence=emotional_valence,
        )

        logger.debug(f"Stored experience: id={episode.id}, content='{content[:50]}...'")

        return episode

    # =========================================================================
    # Consolidation Triggers
    # =========================================================================

    async def trigger_consolidation(
        self,
        mode: str = "light",
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Trigger memory consolidation.

        Maps to biological sleep cycles:
        - light: Quick NREM-like replay (between interactions)
        - deep: Full sleep cycle (end of session)
        - full: Complete consolidation with REM (overnight)

        Args:
            mode: Consolidation mode (light/deep/full)
            force: Force consolidation even if recently run

        Returns:
            Consolidation statistics
        """
        client = self._get_client()

        deep = mode in ("deep", "full")

        result = await client.consolidate(deep=deep)

        logger.info(f"Consolidation triggered: mode={mode}, result={result}")

        return result

    async def train_vae_from_wake_samples(
        self,
        n_samples: int = 100,
        hours: int = 24,
        epochs: int = 5,
    ) -> dict[str, Any] | None:
        """
        P1C: Train VAE from wake samples before consolidation.

        Biological mapping: During wake, the hippocampus collects experiences.
        Before sleep, these experiences train a generative model (VAE) that
        can produce synthetic memories for consolidation replay.

        This implements the wake phase of the wake-sleep algorithm:
        1. Collect recent embeddings from episodic memory
        2. Train VAE to model the distribution of wake experiences
        3. During sleep, VAE generates synthetic samples for replay

        Args:
            n_samples: Maximum samples to collect for training
            hours: Lookback window in hours
            epochs: Training epochs

        Returns:
            Training statistics or None if training unavailable
        """
        client = self._get_client()

        try:
            # Call backend API to train VAE
            result = await client._request(
                "POST",
                "/memory/train-vae",
                json={
                    "n_samples": n_samples,
                    "hours": hours,
                    "epochs": epochs,
                },
            )

            logger.debug(
                f"P1C: VAE training triggered: "
                f"n_samples={n_samples}, hours={hours}, epochs={epochs}"
            )

            return result

        except Exception as e:
            logger.debug(f"P1C: VAE training API not available: {e}")
            return None

    # =========================================================================
    # Statistics and Debugging
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get agent memory client statistics."""
        success_rate = (
            self._successful_tasks / self._total_outcomes
            if self._total_outcomes > 0
            else 0.0
        )

        return {
            "session_id": self.session_id,
            "total_retrievals": self._total_retrievals,
            "total_outcomes": self._total_outcomes,
            "total_credits_assigned": self._total_credits_assigned,
            "successful_tasks": self._successful_tasks,
            "success_rate": success_rate,
            "active_retrievals": len(self._active_retrievals),
            "active_eligibility_traces": self._eligibility.count,
            "dopamine_state": self._dopamine.get_stats(),
            "neuromod_state": self._neuromod_state.to_dict(),
        }

    def get_pending_tasks(self) -> list[str]:
        """Get task IDs that haven't reported outcomes."""
        return list(self._active_retrievals.keys())


# Convenience function for quick access
def create_agent_memory_client(
    base_url: str = "http://localhost:8765",
    session_id: str | None = None,
    **kwargs,
) -> AgentMemoryClient:
    """Create an AgentMemoryClient with default settings."""
    return AgentMemoryClient(
        base_url=base_url,
        session_id=session_id,
        **kwargs,
    )
