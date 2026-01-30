"""
World Weaver Agent for Claude Agent SDK Integration.

Provides a complete agent implementation that:
1. Maintains episodic memory across conversations
2. Learns from task outcomes via three-factor rule
3. Triggers consolidation during idle/session-end
4. Injects relevant context into agent prompts

Based on Boris and CompBio agent recommendations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine
from uuid import uuid4

from ww.sdk.agent_client import AgentMemoryClient, ScoredMemory
from ww.sdk.models import Episode

logger = logging.getLogger(__name__)


class AgentPhase(str, Enum):
    """Agent lifecycle phase (maps to biological states)."""

    IDLE = "idle"  # Between tasks - light consolidation possible
    ENCODING = "encoding"  # Active task - memory formation mode
    RETRIEVAL = "retrieval"  # Gathering context - retrieval mode
    EXECUTING = "executing"  # Running tools - procedural mode
    CONSOLIDATING = "consolidating"  # Sleep/consolidation cycle


@dataclass
class AgentConfig:
    """Configuration for WWAgent."""

    name: str
    model: str = "claude-sonnet-4-5-20250929"
    system_prompt: str | None = None
    tools: list[dict] | None = None

    # Memory settings
    memory_enabled: bool = True
    memory_url: str = "http://localhost:8765"
    memory_api_key: str | None = None

    # Consolidation settings
    consolidation_interval: int = 10  # Messages between consolidations
    idle_consolidation_seconds: float = 300.0  # 5 minutes
    auto_consolidate_on_end: bool = True

    # Context injection
    max_context_memories: int = 5
    min_context_similarity: float = 0.5

    # Learning settings
    base_learning_rate: float = 0.01
    learn_from_outcomes: bool = True


@dataclass
class AgentContext:
    """Context for current agent execution."""

    session_id: str
    phase: AgentPhase = AgentPhase.IDLE
    message_count: int = 0
    current_task_id: str | None = None
    last_activity: datetime = field(default_factory=datetime.now)
    retrieved_memories: list[ScoredMemory] = field(default_factory=list)
    pending_outcomes: dict[str, Any] = field(default_factory=dict)


class WWAgent:
    """
    Claude Agent with World Weaver episodic memory integration.

    This agent:
    1. Retrieves relevant memories before each task
    2. Injects memory context into system prompt
    3. Tracks which memories were used for each task
    4. Reports outcomes to enable learning
    5. Triggers consolidation during idle periods and session end

    Biological mapping (per CompBio agent):
    - Agent task start → Working memory activation (theta encoding)
    - Tool use → Procedural learning (eligibility traces)
    - Task success/failure → Dopamine reward signal
    - Session end → Sleep consolidation trigger
    - Session resume → Memory retrieval + reconsolidation window

    Example:
        agent = WWAgent(
            config=AgentConfig(
                name="code-assistant",
                memory_enabled=True,
            )
        )

        async with agent:
            # First interaction - store memory
            response = await agent.execute(
                messages=[{"role": "user", "content": "How do I fix auth bugs?"}],
                task_id="auth-help"
            )

            # Report outcome
            await agent.report_outcome("auth-help", success=True)

            # Later - retrieves past auth experiences
            response = await agent.execute(
                messages=[{"role": "user", "content": "Another auth issue"}]
            )
    """

    def __init__(
        self,
        config: AgentConfig,
        memory_client: AgentMemoryClient | None = None,
    ):
        """
        Initialize WWAgent.

        Args:
            config: Agent configuration
            memory_client: Optional pre-configured memory client
        """
        self.config = config
        self._memory: AgentMemoryClient | None = memory_client
        self._context: AgentContext | None = None
        self._lifecycle_hooks: dict[str, list[Callable]] = {
            "session_start": [],
            "session_end": [],
            "task_start": [],
            "task_end": [],
            "memory_retrieved": [],
            "outcome_reported": [],
        }
        self._idle_task: asyncio.Task | None = None

    async def __aenter__(self) -> "WWAgent":
        """Start agent session."""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End agent session."""
        await self.end_session()

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    async def start_session(self, session_id: str | None = None):
        """
        Start a new agent session.

        Initializes memory client and sets encoding mode.

        Args:
            session_id: Optional session ID (generated if not provided)
        """
        session_id = session_id or f"{self.config.name}-{uuid4().hex[:8]}"

        # Initialize context
        self._context = AgentContext(
            session_id=session_id,
            phase=AgentPhase.IDLE,
        )

        # Connect memory client
        if self.config.memory_enabled:
            if self._memory is None:
                self._memory = AgentMemoryClient(
                    base_url=self.config.memory_url,
                    session_id=session_id,
                    api_key=self.config.memory_api_key,
                    base_learning_rate=self.config.base_learning_rate,
                )
            await self._memory.connect()

        # Start idle consolidation monitor
        self._start_idle_monitor()

        # Call lifecycle hooks
        await self._call_hooks("session_start", session_id=session_id)

        logger.info(f"Agent session started: {session_id}")

    async def end_session(self):
        """
        End the current session.

        Triggers final consolidation and closes memory client.
        """
        if not self._context:
            return

        # Stop idle monitor
        self._stop_idle_monitor()

        # Report any pending outcomes as neutral
        for task_id in list(self._context.pending_outcomes.keys()):
            logger.warning(f"Task {task_id} ending without outcome, treating as neutral")
            await self.report_outcome(task_id, success=None)

        # Trigger final consolidation
        if self.config.auto_consolidate_on_end and self._memory:
            await self._memory.trigger_consolidation(mode="deep")

        # Close memory client
        if self._memory:
            await self._memory.close()
            self._memory = None

        # Call lifecycle hooks
        await self._call_hooks("session_end", session_id=self._context.session_id)

        logger.info(f"Agent session ended: {self._context.session_id}")
        self._context = None

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def execute(
        self,
        messages: list[dict[str, Any]],
        task_id: str | None = None,
        include_memory_context: bool = True,
    ) -> dict[str, Any]:
        """
        Execute agent with memory-augmented context.

        Args:
            messages: Conversation messages
            task_id: Optional task identifier (generated if not provided)
            include_memory_context: Whether to retrieve and inject memories

        Returns:
            Agent response with metadata
        """
        if not self._context:
            raise RuntimeError("Session not started. Use 'async with' or call start_session()")

        task_id = task_id or f"task-{uuid4().hex[:8]}"
        self._context.current_task_id = task_id
        self._context.phase = AgentPhase.ENCODING
        self._context.last_activity = datetime.now()

        # Call task start hooks
        await self._call_hooks("task_start", task_id=task_id, messages=messages)

        # Retrieve relevant memories
        memory_context = ""
        if include_memory_context and self._memory and messages:
            self._context.phase = AgentPhase.RETRIEVAL
            memory_context = await self._retrieve_context(messages, task_id)

        # Build augmented system prompt
        system_prompt = self._build_system_prompt(memory_context)

        # Track for outcome reporting
        self._context.pending_outcomes[task_id] = {
            "messages": messages,
            "memory_context": memory_context,
            "start_time": datetime.now(),
        }

        # Execute with Claude (placeholder - actual implementation depends on SDK)
        self._context.phase = AgentPhase.EXECUTING
        response = await self._execute_with_claude(
            messages=messages,
            system_prompt=system_prompt,
            task_id=task_id,
        )

        # Update message count
        self._context.message_count += 1

        # Check if consolidation needed
        if self._context.message_count % self.config.consolidation_interval == 0:
            await self._trigger_periodic_consolidation()

        # Call task end hooks
        await self._call_hooks("task_end", task_id=task_id, response=response)

        self._context.phase = AgentPhase.IDLE
        self._context.last_activity = datetime.now()

        return response

    async def _retrieve_context(
        self,
        messages: list[dict[str, Any]],
        task_id: str,
    ) -> str:
        """Retrieve relevant memories and format as context."""
        if not self._memory:
            return ""

        # Extract query from last user message
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    query = content
                elif isinstance(content, list):
                    # Handle content blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            query = block.get("text", "")
                            break
                break

        if not query:
            return ""

        # Retrieve memories
        memories = await self._memory.retrieve_for_task(
            task_id=task_id,
            query=query,
            limit=self.config.max_context_memories,
            min_similarity=self.config.min_context_similarity,
        )

        self._context.retrieved_memories = memories

        # Call hooks
        await self._call_hooks("memory_retrieved", task_id=task_id, memories=memories)

        # Format as context string
        if not memories:
            return ""

        context_parts = ["## Relevant Past Experiences\n"]
        for i, mem in enumerate(memories, 1):
            context_parts.append(
                f"{i}. [{mem.episode.outcome}] {mem.episode.content[:200]}..."
                if len(mem.episode.content) > 200
                else f"{i}. [{mem.episode.outcome}] {mem.episode.content}"
            )

        return "\n".join(context_parts)

    def _build_system_prompt(self, memory_context: str) -> str:
        """Build system prompt with memory context."""
        parts = []

        # Base system prompt
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)

        # Memory context
        if memory_context:
            parts.append(
                "\n\n---\n"
                "You have access to the following relevant past experiences. "
                "Use these to inform your approach when applicable:\n\n"
                f"{memory_context}"
            )

        return "\n".join(parts) if parts else ""

    async def _execute_with_claude(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        task_id: str,
    ) -> dict[str, Any]:
        """
        Execute with Claude API.

        This is a placeholder that would integrate with Claude Agent SDK.
        In practice, this would call ClaudeSDKClient or similar.
        """
        # For now, return a structured response
        # Actual implementation would use Claude Agent SDK
        return {
            "task_id": task_id,
            "session_id": self._context.session_id if self._context else None,
            "message_count": self._context.message_count if self._context else 0,
            "memory_context_used": bool(system_prompt),
            "status": "completed",
            # Would include actual Claude response here
        }

    # =========================================================================
    # Outcome Reporting
    # =========================================================================

    async def report_outcome(
        self,
        task_id: str,
        success: bool | None = None,
        partial_credit: float | None = None,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """
        Report task outcome for learning.

        Args:
            task_id: Task identifier from execute()
            success: True for success, False for failure, None for neutral
            partial_credit: Float 0-1 for partial success
            feedback: Optional feedback for logging

        Returns:
            Credit assignment results
        """
        if not self._context:
            raise RuntimeError("Session not started")

        # Remove from pending
        task_info = self._context.pending_outcomes.pop(task_id, None)

        # Report to memory client
        result = {}
        if self._memory and self.config.learn_from_outcomes:
            credit_result = await self._memory.report_task_outcome(
                task_id=task_id,
                success=success,
                partial_credit=partial_credit,
                feedback=feedback,
            )
            result = {
                "credited": credit_result.credited,
                "reconsolidated": credit_result.reconsolidated,
                "total_lr": credit_result.total_lr_applied,
            }

        # Store the interaction as an experience
        if self._memory and task_info:
            outcome_str = "success" if success else ("failure" if success is False else "neutral")
            content = f"Task: {task_info.get('messages', [{}])[-1].get('content', '')[:200]}"
            await self._memory.store_experience(
                content=content,
                outcome=outcome_str,
                importance=0.7 if success else 0.5,
            )

        # Call hooks
        await self._call_hooks(
            "outcome_reported",
            task_id=task_id,
            success=success,
            result=result,
        )

        logger.info(f"Outcome reported for task={task_id}: success={success}")

        return result

    # =========================================================================
    # Consolidation
    # =========================================================================

    async def _trigger_periodic_consolidation(self):
        """Trigger light consolidation between tasks."""
        if self._memory:
            logger.debug("Triggering periodic consolidation")
            self._context.phase = AgentPhase.CONSOLIDATING
            await self._memory.trigger_consolidation(mode="light")
            self._context.phase = AgentPhase.IDLE

    def _start_idle_monitor(self):
        """Start background task to monitor for idle consolidation."""
        if self._idle_task is not None:
            return

        async def monitor():
            while self._context is not None:
                await asyncio.sleep(60)  # Check every minute

                if self._context is None:
                    break

                idle_duration = (datetime.now() - self._context.last_activity).total_seconds()

                if (
                    idle_duration > self.config.idle_consolidation_seconds
                    and self._context.phase == AgentPhase.IDLE
                    and self._memory
                ):
                    logger.info(f"Idle consolidation triggered after {idle_duration:.0f}s")
                    self._context.phase = AgentPhase.CONSOLIDATING
                    await self._memory.trigger_consolidation(mode="light")
                    self._context.phase = AgentPhase.IDLE
                    self._context.last_activity = datetime.now()

        self._idle_task = asyncio.create_task(monitor())

    def _stop_idle_monitor(self):
        """Stop idle monitor task."""
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    def on(
        self,
        event: str,
        handler: Callable[..., Coroutine[Any, Any, None]],
    ):
        """
        Register a lifecycle hook.

        Events:
        - session_start: Called when session begins
        - session_end: Called when session ends
        - task_start: Called before task execution
        - task_end: Called after task execution
        - memory_retrieved: Called after memories are retrieved
        - outcome_reported: Called after outcome is reported

        Args:
            event: Event name
            handler: Async handler function
        """
        if event not in self._lifecycle_hooks:
            raise ValueError(f"Unknown event: {event}")
        self._lifecycle_hooks[event].append(handler)

    async def _call_hooks(self, event: str, **kwargs):
        """Call all registered hooks for an event."""
        for handler in self._lifecycle_hooks.get(event, []):
            try:
                await handler(**kwargs)
            except Exception as e:
                logger.error(f"Hook error for {event}: {e}")

    # =========================================================================
    # Direct Memory Access
    # =========================================================================

    async def store_memory(
        self,
        content: str,
        outcome: str = "neutral",
        importance: float = 0.5,
        project: str | None = None,
    ) -> Episode | None:
        """
        Directly store a memory.

        Args:
            content: Memory content
            outcome: Outcome category
            importance: Importance score 0-1
            project: Optional project context

        Returns:
            Created episode or None if memory disabled
        """
        if not self._memory:
            return None

        return await self._memory.store_experience(
            content=content,
            outcome=outcome,
            importance=importance,
            project=project,
        )

    async def recall_memories(
        self,
        query: str,
        limit: int = 5,
    ) -> list[Episode]:
        """
        Directly recall memories.

        Args:
            query: Search query
            limit: Maximum memories

        Returns:
            List of episodes
        """
        if not self._memory:
            return []

        task_id = f"recall-{uuid4().hex[:8]}"
        memories = await self._memory.retrieve_for_task(
            task_id=task_id,
            query=query,
            limit=limit,
        )

        # Don't track this for outcome - immediate cleanup
        self._memory._active_retrievals.pop(task_id, None)

        return [m.episode for m in memories]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "name": self.config.name,
            "session_id": self._context.session_id if self._context else None,
            "phase": self._context.phase.value if self._context else None,
            "message_count": self._context.message_count if self._context else 0,
            "pending_outcomes": len(self._context.pending_outcomes) if self._context else 0,
        }

        if self._memory:
            stats["memory"] = self._memory.get_stats()

        return stats
