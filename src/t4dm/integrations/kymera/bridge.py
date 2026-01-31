"""
Voice Memory Bridge - Connects Kymera Voice to World Weaver memory.

Handles:
- Deciding when to store voice interactions as episodes
- Retrieving relevant context for voice queries
- Managing voice conversation state in memory
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from t4dm.core.memory_gate import GateContext, MemoryGate, StorageDecision
from t4dm.core.privacy_filter import PrivacyFilter
from t4dm.core.types import EpisodeContext, Outcome

logger = logging.getLogger(__name__)


@dataclass
class VoiceContext:
    """Context for a voice interaction."""
    session_id: str
    project: str | None = None
    cwd: str | None = None
    active_file: str | None = None
    conversation_id: str | None = None
    turn_number: int = 0
    is_voice: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "project": self.project,
            "cwd": self.cwd,
            "active_file": self.active_file,
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "is_voice": self.is_voice,
        }


@dataclass
class MemoryContext:
    """Retrieved memory context for a query."""
    episodes: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)
    personal_context: str | None = None

    def to_prompt(self) -> str:
        """Format as prompt injection."""
        parts = []

        if self.episodes:
            parts.append("## Recent Relevant History")
            for ep in self.episodes[:5]:
                parts.append(f"- {ep.get('content', '')[:200]}")

        if self.entities:
            parts.append("\n## Relevant Knowledge")
            for ent in self.entities[:10]:
                parts.append(f"- **{ent.get('name', '')}**: {ent.get('summary', '')[:100]}")

        if self.skills:
            parts.append("\n## Available Procedures")
            for skill in self.skills[:3]:
                parts.append(f"- {skill.get('name', '')}: {skill.get('description', '')[:100]}")

        if self.personal_context:
            parts.append(f"\n## Current Context\n{self.personal_context}")

        return "\n".join(parts) if parts else ""


class VoiceMemoryBridge:
    """
    Bridge between Kymera Voice and World Weaver memory.

    Responsibilities:
    1. Filter voice input through PrivacyFilter
    2. Decide whether to store via MemoryGate
    3. Retrieve relevant context for queries
    4. Batch and store conversation episodes
    """

    def __init__(
        self,
        ww_client: Any,  # World Weaver MCP client
        memory_gate: MemoryGate | None = None,
        privacy_filter: PrivacyFilter | None = None,
    ):
        """
        Initialize voice memory bridge.

        Args:
            ww_client: World Weaver MCP client for memory operations
            memory_gate: Memory gate for storage decisions (default: create new)
            privacy_filter: Privacy filter for redaction (default: create new)
        """
        self.ww = ww_client
        self.gate = memory_gate or MemoryGate(
            store_threshold=0.4,
            buffer_threshold=0.2,
            voice_mode_adjustments=True,
        )
        self.privacy = privacy_filter or PrivacyFilter()

        # Temporal batcher for grouping voice turns
        self.batcher = TemporalBatcher(
            batch_window_minutes=2,
            max_batch_size=10,
        )

        # Track recent entities for context
        self._recent_entities: list[str] = []

        logger.info("VoiceMemoryBridge initialized")

    async def on_user_speech(
        self,
        text: str,
        context: VoiceContext,
        store_immediately: bool = False,
    ) -> UUID | None:
        """
        Process user speech and potentially store as episode.

        Args:
            text: Transcribed user speech
            context: Voice interaction context
            store_immediately: Force immediate storage (bypass gate)

        Returns:
            Episode ID if stored, None otherwise
        """
        # Step 1: Privacy filter
        filtered = self.privacy.filter(text)

        if filtered.blocked:
            logger.debug("User speech blocked by privacy filter")
            return None

        content = filtered.content

        # Step 2: Memory gate decision
        gate_ctx = GateContext(
            session_id=context.session_id,
            project=context.project,
            cwd=context.cwd,
            recent_entities=self._recent_entities,
            is_voice=True,
        )

        if store_immediately:
            decision_type = StorageDecision.STORE
            importance = 0.7
        else:
            decision = self.gate.evaluate(content, gate_ctx)
            decision_type = decision.decision
            importance = decision.suggested_importance

        # Step 3: Act on decision
        if decision_type == StorageDecision.STORE:
            return await self._store_episode(content, context, importance)

        if decision_type == StorageDecision.BUFFER:
            # Add to batch
            batch_key = f"{context.session_id}:{context.conversation_id or 'default'}"
            batched = self.batcher.add(batch_key, content)

            if batched:
                # Batch complete - store combined episode
                return await self._store_episode(batched, context, importance * 0.8)

        # SKIP - don't store
        return None

    async def on_assistant_response(
        self,
        text: str,
        spoken_text: str,
        context: VoiceContext,
        was_action: bool = False,
    ) -> UUID | None:
        """
        Process assistant response.

        Generally we don't store assistant responses unless:
        - They contain learned information ("I'll remember that")
        - They completed an action successfully
        - User explicitly asked to remember something

        Args:
            text: Full assistant response
            spoken_text: What was actually spoken (may be summarized)
            context: Voice interaction context
            was_action: Whether response was from an action

        Returns:
            Episode ID if stored, None otherwise
        """
        # Check for memory triggers
        memory_triggers = [
            "i'll remember",
            "noted",
            "i've stored",
            "i've learned",
            "saving that",
        ]

        should_store = any(trigger in text.lower() for trigger in memory_triggers)
        should_store = should_store or was_action

        if should_store:
            filtered = self.privacy.filter(text)
            if not filtered.blocked:
                return await self._store_episode(
                    f"Assistant: {filtered.content}",
                    context,
                    importance=0.5,
                    outcome=Outcome.SUCCESS if was_action else Outcome.NEUTRAL,
                )

        return None

    async def on_conversation_end(
        self,
        context: VoiceContext,
        summary: str | None = None,
    ) -> UUID | None:
        """
        Handle end of voice conversation.

        Flushes any batched content and optionally stores conversation summary.

        Args:
            context: Voice interaction context
            summary: Optional conversation summary

        Returns:
            Episode ID if summary stored, None otherwise
        """
        # Flush batched content
        batched_items = self.batcher.flush_all()
        for batch_key, content in batched_items:
            await self._store_episode(content, context, importance=0.5)

        # Store summary if provided
        if summary:
            return await self._store_episode(
                summary,
                context,
                importance=0.6,
                outcome=Outcome.SUCCESS,
            )

        return None

    async def get_relevant_context(
        self,
        query: str,
        context: VoiceContext,
        include_personal: bool = True,
    ) -> MemoryContext:
        """
        Get relevant memory context for a voice query.

        Args:
            query: The user's query/request
            context: Current voice context
            include_personal: Include personal data (calendar, etc.)

        Returns:
            MemoryContext with relevant memories
        """
        memory_ctx = MemoryContext()

        try:
            # Recall recent episodes
            episodes_result = await self.ww.call_tool(
                "mcp__ww-memory__recall_episodes",
                {
                    "query": query,
                    "limit": 5,
                    "time_filter": {"after": "7 days ago"},
                }
            )
            memory_ctx.episodes = episodes_result.get("episodes", [])

            # Semantic recall for entities
            entities_result = await self.ww.call_tool(
                "mcp__ww-memory__semantic_recall",
                {
                    "query": query,
                    "limit": 10,
                    "include_relationships": True,
                }
            )
            memory_ctx.entities = entities_result.get("entities", [])

            # Update recent entities for gate
            self._recent_entities = [
                e.get("name", "") for e in memory_ctx.entities[:10]
            ]

            # Recall applicable skills
            skills_result = await self.ww.call_tool(
                "mcp__ww-memory__recall_skill",
                {
                    "query": query,
                    "limit": 3,
                    "check_preconditions": True,
                    "context": context.to_dict(),
                }
            )
            memory_ctx.skills = skills_result.get("skills", [])

            # Get personal context if requested
            if include_personal:
                memory_ctx.personal_context = await self._get_personal_context(context)

        except Exception as e:
            logger.error(f"Error getting memory context: {e}")

        return memory_ctx

    async def store_explicit_memory(
        self,
        content: str,
        context: VoiceContext,
    ) -> UUID:
        """
        Store explicit "remember this" request.

        Called when user says "remember that..." - always stores.

        Args:
            content: What to remember
            context: Voice interaction context

        Returns:
            Episode ID
        """
        filtered = self.privacy.filter(content)

        if filtered.blocked:
            raise ValueError("Cannot store - content blocked by privacy filter")

        return await self._store_episode(
            f"User asked to remember: {filtered.content}",
            context,
            importance=0.9,
        )

    async def recall_explicit(
        self,
        query: str,
        context: VoiceContext,
    ) -> list[dict]:
        """
        Handle explicit "what do you remember about..." query.

        Args:
            query: What to recall
            context: Voice interaction context

        Returns:
            List of relevant memories
        """
        result = await self.ww.call_tool(
            "mcp__ww-memory__recall_episodes",
            {
                "query": query,
                "limit": 10,
            }
        )
        return result.get("episodes", [])

    async def forget(
        self,
        query: str,
        context: VoiceContext,
    ) -> int:
        """
        Handle "forget about..." request.

        Args:
            query: What to forget
            context: Voice interaction context

        Returns:
            Number of memories affected
        """
        # This would call a delete/archive operation
        # For now, just log - actual deletion needs careful handling
        logger.info(f"Forget request: {query}")
        return 0  # Placeholder

    async def _store_episode(
        self,
        content: str,
        context: VoiceContext,
        importance: float,
        outcome: Outcome = Outcome.NEUTRAL,
    ) -> UUID:
        """Store an episode in World Weaver."""
        episode_context = EpisodeContext(
            project=context.project,
            cwd=context.cwd,
            task=f"voice_conversation:{context.conversation_id or 'default'}",
            files_touched=[context.active_file] if context.active_file else [],
        )

        result = await self.ww.call_tool(
            "mcp__ww-memory__store_episode",
            {
                "content": content,
                "session_id": context.session_id,
                "context": episode_context.model_dump(),
                "outcome": outcome.value,
                "emotional_valence": importance,
            }
        )

        episode_id = UUID(result["episode_id"])
        logger.debug(f"Stored episode {episode_id}: {content[:50]}...")
        return episode_id

    async def _get_personal_context(self, context: VoiceContext) -> str:
        """Get personal context (calendar, tasks, etc.)."""
        parts = []

        try:
            # Get today's events
            from datetime import datetime

            now = datetime.now()
            events_result = await self.ww.call_tool(
                "mcp__google-workspace__calendar_list_events",
                {
                    "date": now.strftime("%Y-%m-%d"),
                    "days": 1,
                    "maxResults": 5,
                }
            )

            events = events_result.get("items", [])
            if events:
                parts.append(f"Today's events: {len(events)}")
                for e in events[:3]:
                    parts.append(f"- {e.get('summary', 'Untitled')}")

        except Exception as e:
            logger.debug(f"Could not get personal context: {e}")

        return "\n".join(parts) if parts else ""


class TemporalBatcher:
    """
    Batches voice content over time windows.

    Groups multiple voice turns into coherent episodes
    for more meaningful storage.
    """

    def __init__(
        self,
        batch_window_minutes: int = 2,
        max_batch_size: int = 10,
    ):
        self.batch_window = batch_window_minutes * 60  # seconds
        self.max_batch_size = max_batch_size
        self._batches: dict[str, list[tuple[datetime, str]]] = {}

    def add(self, batch_key: str, content: str) -> str | None:
        """
        Add content to batch, return combined content if batch ready.

        Args:
            batch_key: Key for grouping related content
            content: Content to add

        Returns:
            Combined batch content if ready, None otherwise
        """
        now = datetime.now()

        if batch_key not in self._batches:
            self._batches[batch_key] = []

        batch = self._batches[batch_key]

        # Check if should flush first
        if batch:
            first_time = batch[0][0]
            elapsed = (now - first_time).total_seconds()

            if elapsed > self.batch_window or len(batch) >= self.max_batch_size:
                result = self._combine_batch(batch)
                self._batches[batch_key] = [(now, content)]
                return result

        batch.append((now, content))
        return None

    def flush_all(self) -> list[tuple[str, str]]:
        """Flush all batches, return list of (key, content)."""
        results = []

        for key, batch in self._batches.items():
            if batch:
                combined = self._combine_batch(batch)
                if combined:
                    results.append((key, combined))

        self._batches.clear()
        return results

    def _combine_batch(self, batch: list[tuple[datetime, str]]) -> str:
        """Combine batch items into single content."""
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for _, content in batch:
            if content not in seen:
                seen.add(content)
                unique.append(content)

        return " | ".join(unique)
