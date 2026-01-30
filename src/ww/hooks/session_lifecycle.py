"""
Session Lifecycle Hooks for Claude Agent SDK Integration.

Provides hooks for:
- SessionStart: Load context, initialize encoding mode
- SessionEnd: Persist session, trigger consolidation
- TaskComplete: Learn from outcomes

Based on CompBio agent biological mapping:
- Session start → Working memory activation (theta encoding phase)
- Session end → Sleep consolidation trigger
- Task complete → Dopamine reward signal
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine
from uuid import uuid4

from ww.hooks.base import Hook, HookContext, HookPhase, HookPriority
from ww.sdk.agent_client import AgentMemoryClient

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Context for a session lifecycle."""

    session_id: str
    project: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    task_count: int = 0
    successful_tasks: int = 0
    memories_retrieved: int = 0
    memories_stored: int = 0
    consolidations: int = 0
    vae_training_sessions: int = 0  # P1C: Track VAE training
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStartHook(Hook):
    """
    Hook for session initialization.

    Biological mapping: Working memory activation + theta encoding phase.

    Actions:
    1. Initialize memory client connection
    2. Set neuromodulator state to encoding mode (high ACh)
    3. Load relevant context from past sessions
    4. Pre-warm eligibility traces
    """

    def __init__(
        self,
        memory_client: AgentMemoryClient | None = None,
        load_context: bool = True,
        context_query: str | None = None,
        max_context_items: int = 5,
        on_context_loaded: Callable[[list], Coroutine] | None = None,
    ):
        """
        Initialize session start hook.

        Args:
            memory_client: Memory client for context loading
            load_context: Whether to load past context automatically
            context_query: Custom query for context loading
            max_context_items: Maximum context items to load
            on_context_loaded: Callback after context is loaded
        """
        super().__init__(
            name="session_start",
            priority=HookPriority.CRITICAL,
            enabled=True,
        )
        self._memory = memory_client
        self._load_context = load_context
        self._context_query = context_query
        self._max_context_items = max_context_items
        self._on_context_loaded = on_context_loaded
        self._sessions: dict[str, SessionContext] = {}

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute session start actions.

        Args:
            context: Hook context with session information

        Returns:
            Modified context with loaded memories
        """
        session_id = context.session_id or f"session-{uuid4().hex[:8]}"

        # Create session context
        session_ctx = SessionContext(
            session_id=session_id,
            project=context.metadata.get("project"),
        )
        self._sessions[session_id] = session_ctx

        logger.info(f"Session started: {session_id}")

        # Load relevant context if enabled
        if self._load_context and self._memory:
            try:
                query = self._context_query or "recent session context and patterns"

                memories = await self._memory.retrieve_for_task(
                    task_id=f"session-init-{session_id}",
                    query=query,
                    limit=self._max_context_items,
                )

                # Store in context output
                context.output_data = context.output_data or {}
                context.output_data["loaded_memories"] = [
                    {
                        "id": str(m.episode.id),
                        "content": m.episode.content,
                        "outcome": m.episode.outcome,
                        "score": m.combined_score,
                    }
                    for m in memories
                ]

                session_ctx.memories_retrieved = len(memories)

                if self._on_context_loaded:
                    await self._on_context_loaded(memories)

                logger.debug(f"Loaded {len(memories)} context items for session {session_id}")

            except Exception as e:
                logger.error(f"Failed to load session context: {e}")

        # Set session ID in context
        context.session_id = session_id
        context.metadata["session_context"] = session_ctx

        return context

    def get_session(self, session_id: str) -> SessionContext | None:
        """Get session context by ID."""
        return self._sessions.get(session_id)


class SessionEndHook(Hook):
    """
    Hook for session termination.

    Biological mapping: Sleep consolidation trigger.

    Actions:
    1. Store session summary
    2. Report any pending outcomes
    3. P1C: Train VAE from wake samples
    4. Trigger deep consolidation
    5. Persist session statistics
    """

    def __init__(
        self,
        memory_client: AgentMemoryClient | None = None,
        auto_consolidate: bool = True,
        consolidation_mode: str = "deep",
        store_summary: bool = True,
        train_vae_before_sleep: bool = True,  # P1C: VAE training flag
        on_session_end: Callable[[SessionContext], Coroutine] | None = None,
    ):
        """
        Initialize session end hook.

        Args:
            memory_client: Memory client for consolidation
            auto_consolidate: Whether to trigger consolidation
            consolidation_mode: Consolidation intensity (light/deep/full)
            store_summary: Whether to store session summary
            train_vae_before_sleep: P1C: Train VAE from wake samples before sleep
            on_session_end: Callback after session ends
        """
        super().__init__(
            name="session_end",
            priority=HookPriority.LOW,  # Run after other cleanup
            enabled=True,
        )
        self._memory = memory_client
        self._auto_consolidate = auto_consolidate
        self._consolidation_mode = consolidation_mode
        self._store_summary = store_summary
        self._train_vae_before_sleep = train_vae_before_sleep
        self._on_session_end = on_session_end

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute session end actions.

        Args:
            context: Hook context with session information

        Returns:
            Modified context with session statistics
        """
        session_ctx: SessionContext | None = context.metadata.get("session_context")

        if not session_ctx:
            logger.warning("No session context found for session end")
            return context

        session_ctx.end_time = datetime.now()

        logger.info(
            f"Session ending: {session_ctx.session_id}, "
            f"tasks={session_ctx.task_count}, "
            f"success_rate={session_ctx.successful_tasks / max(1, session_ctx.task_count):.1%}"
        )

        if self._memory:
            # Store session summary
            if self._store_summary:
                try:
                    duration = (session_ctx.end_time - session_ctx.start_time).total_seconds()
                    summary = (
                        f"Session {session_ctx.session_id}: "
                        f"{session_ctx.task_count} tasks, "
                        f"{session_ctx.successful_tasks} successful, "
                        f"{duration:.0f}s duration"
                    )

                    await self._memory.store_experience(
                        content=summary,
                        outcome="success" if session_ctx.successful_tasks > 0 else "neutral",
                        importance=0.6,
                        project=session_ctx.project,
                    )
                    session_ctx.memories_stored += 1

                except Exception as e:
                    logger.error(f"Failed to store session summary: {e}")

            # P1C: Train VAE from wake samples before consolidation
            if self._train_vae_before_sleep:
                try:
                    # Access sleep manager to train VAE
                    vae_stats = await self._memory.train_vae_from_wake_samples()
                    if vae_stats:
                        session_ctx.vae_training_sessions += 1
                        logger.info(
                            f"P1C: VAE trained before sleep: "
                            f"loss={vae_stats.get('final_loss', 'N/A')}, "
                            f"samples={vae_stats.get('samples_trained', 0)}"
                        )

                        # Store VAE stats in output
                        context.output_data = context.output_data or {}
                        context.output_data["vae_training"] = vae_stats

                except AttributeError:
                    logger.debug("P1C: VAE training not available (method not implemented)")
                except Exception as e:
                    logger.warning(f"P1C: VAE training failed: {e}")

            # Trigger consolidation
            if self._auto_consolidate:
                try:
                    await self._memory.trigger_consolidation(mode=self._consolidation_mode)
                    session_ctx.consolidations += 1
                    logger.info(f"Consolidation triggered: mode={self._consolidation_mode}")

                except Exception as e:
                    logger.error(f"Failed to trigger consolidation: {e}")

        # Callback
        if self._on_session_end:
            await self._on_session_end(session_ctx)

        # Store statistics in output
        context.output_data = context.output_data or {}
        context.output_data["session_stats"] = {
            "session_id": session_ctx.session_id,
            "duration_seconds": (session_ctx.end_time - session_ctx.start_time).total_seconds(),
            "task_count": session_ctx.task_count,
            "successful_tasks": session_ctx.successful_tasks,
            "memories_retrieved": session_ctx.memories_retrieved,
            "memories_stored": session_ctx.memories_stored,
            "consolidations": session_ctx.consolidations,
            "vae_training_sessions": session_ctx.vae_training_sessions,  # P1C
        }

        return context


class TaskOutcomeHook(Hook):
    """
    Hook for task outcome reporting.

    Biological mapping: Dopamine reward signal (VTA burst/pause).

    Actions:
    1. Compute reward prediction error
    2. Propagate credit to retrieved memories
    3. Update eligibility traces
    4. Store task experience
    """

    def __init__(
        self,
        memory_client: AgentMemoryClient | None = None,
        auto_store_experience: bool = True,
        on_outcome: Callable[[str, bool, dict], Coroutine] | None = None,
    ):
        """
        Initialize task outcome hook.

        Args:
            memory_client: Memory client for learning
            auto_store_experience: Whether to store task as experience
            on_outcome: Callback after outcome is processed
        """
        super().__init__(
            name="task_outcome",
            priority=HookPriority.HIGH,  # Before cleanup
            enabled=True,
        )
        self._memory = memory_client
        self._auto_store_experience = auto_store_experience
        self._on_outcome = on_outcome

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute task outcome processing.

        Args:
            context: Hook context with task outcome

        Returns:
            Modified context with learning results
        """
        task_id = context.metadata.get("task_id")
        success = context.metadata.get("success")
        partial_credit = context.metadata.get("partial_credit")
        task_content = context.metadata.get("task_content", "")

        if task_id is None:
            logger.warning("No task_id in outcome context")
            return context

        logger.info(f"Task outcome: {task_id}, success={success}")

        credit_result = None

        if self._memory:
            # Report outcome for credit assignment
            try:
                credit_result = await self._memory.report_task_outcome(
                    task_id=task_id,
                    success=success,
                    partial_credit=partial_credit,
                    feedback=context.metadata.get("feedback"),
                )

                logger.debug(
                    f"Credit assigned: {credit_result.credited} memories, "
                    f"LR={credit_result.total_lr_applied:.4f}"
                )

            except Exception as e:
                logger.error(f"Failed to report outcome: {e}")

            # Store task experience
            if self._auto_store_experience and task_content:
                try:
                    outcome_str = "success" if success else ("failure" if success is False else "neutral")
                    await self._memory.store_experience(
                        content=task_content[:500],  # Truncate long content
                        outcome=outcome_str,
                        importance=0.7 if success else 0.4,
                    )

                except Exception as e:
                    logger.error(f"Failed to store task experience: {e}")

        # Update session context
        session_ctx: SessionContext | None = context.metadata.get("session_context")
        if session_ctx:
            session_ctx.task_count += 1
            if success:
                session_ctx.successful_tasks += 1

        # Callback
        if self._on_outcome:
            await self._on_outcome(
                task_id,
                success,
                credit_result.__dict__ if credit_result else {},
            )

        # Store results in output
        context.output_data = context.output_data or {}
        if credit_result:
            context.output_data["credit_result"] = {
                "credited": credit_result.credited,
                "reconsolidated": credit_result.reconsolidated,
                "total_lr": credit_result.total_lr_applied,
            }

        return context


class IdleConsolidationHook(Hook):
    """
    Hook for idle-triggered consolidation.

    Biological mapping: NREM-like consolidation during inactivity.

    Triggers light consolidation when agent is idle for extended period.
    """

    def __init__(
        self,
        memory_client: AgentMemoryClient | None = None,
        idle_threshold_seconds: float = 300.0,  # 5 minutes
        consolidation_mode: str = "light",
    ):
        """
        Initialize idle consolidation hook.

        Args:
            memory_client: Memory client for consolidation
            idle_threshold_seconds: Seconds of inactivity before consolidation
            consolidation_mode: Consolidation intensity
        """
        super().__init__(
            name="idle_consolidation",
            priority=HookPriority.LOW,
            enabled=True,
        )
        self._memory = memory_client
        self._idle_threshold = idle_threshold_seconds
        self._consolidation_mode = consolidation_mode
        self._last_activity: datetime | None = None
        self._consolidation_count = 0

    async def execute(self, context: HookContext) -> HookContext:
        """
        Check for idle consolidation trigger.

        Args:
            context: Hook context with timing information

        Returns:
            Modified context
        """
        current_time = datetime.now()

        if self._last_activity:
            idle_duration = (current_time - self._last_activity).total_seconds()

            if idle_duration >= self._idle_threshold and self._memory:
                logger.info(f"Idle consolidation triggered after {idle_duration:.0f}s")

                try:
                    await self._memory.trigger_consolidation(mode=self._consolidation_mode)
                    self._consolidation_count += 1

                    context.output_data = context.output_data or {}
                    context.output_data["idle_consolidation"] = {
                        "triggered": True,
                        "idle_duration": idle_duration,
                        "mode": self._consolidation_mode,
                    }

                except Exception as e:
                    logger.error(f"Idle consolidation failed: {e}")

        self._last_activity = current_time
        return context

    def update_activity(self):
        """Update last activity timestamp."""
        self._last_activity = datetime.now()


# Convenience functions for hook creation


def create_session_hooks(
    memory_client: AgentMemoryClient,
    auto_consolidate: bool = True,
    load_context: bool = True,
    train_vae: bool = True,  # P1C: VAE training flag
) -> tuple[SessionStartHook, SessionEndHook, TaskOutcomeHook]:
    """
    Create a complete set of session lifecycle hooks.

    Args:
        memory_client: Memory client for all hooks
        auto_consolidate: Whether to auto-consolidate on session end
        load_context: Whether to load context on session start
        train_vae: P1C: Whether to train VAE before consolidation

    Returns:
        Tuple of (start_hook, end_hook, outcome_hook)
    """
    start_hook = SessionStartHook(
        memory_client=memory_client,
        load_context=load_context,
    )

    end_hook = SessionEndHook(
        memory_client=memory_client,
        auto_consolidate=auto_consolidate,
        train_vae_before_sleep=train_vae,  # P1C
    )

    outcome_hook = TaskOutcomeHook(
        memory_client=memory_client,
        auto_store_experience=True,
    )

    return start_hook, end_hook, outcome_hook
