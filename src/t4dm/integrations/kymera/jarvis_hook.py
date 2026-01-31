"""
Jarvis Hook - Integration point for Kymera Voice's Jarvis interface.

This module provides hooks that can be injected into the existing
Kymera Voice Jarvis class to add World Weaver memory capabilities.

Usage in kymera-voice/src/kymera_voice/core/jarvis.py:

    from t4dm.integrations.kymera import JarvisMemoryHook

    class Jarvis:
        def __init__(self, ...):
            # ... existing init ...
            self.memory_hook = JarvisMemoryHook.create_async(ww_url="http://localhost:8765")

        async def process_speech(self, text: str):
            # Get memory-enhanced context
            if self.memory_hook:
                context = await self.memory_hook.enhance_context(text, self.session_id)
                # Use context in Claude prompt

            # ... existing processing ...

            # Store interaction
            if self.memory_hook:
                await self.memory_hook.on_response(text, response, self.session_id)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

logger = logging.getLogger(__name__)


class MCPClientProtocol(Protocol):
    """Protocol for MCP client interface."""
    async def call_tool(self, name: str, params: dict) -> dict: ...


@dataclass
class EnhancedContext:
    """Enhanced context for Claude prompt."""
    system_prompt_addition: str
    relevant_memories: list[dict]
    personal_context: str | None
    proactive_message: str | None


class JarvisMemoryHook:
    """
    Memory hook for Jarvis voice interface.

    Provides methods to enhance voice interactions with World Weaver memory.
    """

    def __init__(
        self,
        ww_client: MCPClientProtocol,
        session_id: str | None = None,
    ):
        """
        Initialize Jarvis memory hook.

        Args:
            ww_client: World Weaver MCP client
            session_id: Optional persistent session ID
        """
        self.ww = ww_client
        self.session_id = session_id or f"jarvis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._conversation_id: str | None = None
        self._turn_count = 0
        self._active = True

        # Import components lazily to avoid circular imports
        self._executor = None
        self._bridge = None

    @classmethod
    async def create_async(
        cls,
        ww_url: str = "http://localhost:8765",
        session_id: str | None = None,
    ) -> "JarvisMemoryHook":
        """
        Async factory to create hook with connected MCP client.

        DEPRECATED: MCP client has been removed. Use create_with_memory_api() instead.

        Args:
            ww_url: World Weaver MCP server URL (ignored)
            session_id: Optional session ID

        Returns:
            Initialized JarvisMemoryHook

        Raises:
            NotImplementedError: MCP client is no longer available
        """
        raise NotImplementedError(
            "MCP client has been removed in v0.2.0. "
            "Use JarvisMemoryHook.create_with_memory_api(session_id) instead, "
            "which uses the direct Python API."
        )

    @classmethod
    def create_with_client(
        cls,
        ww_client: MCPClientProtocol,
        session_id: str | None = None,
    ) -> "JarvisMemoryHook":
        """
        Create hook with existing MCP client.

        Args:
            ww_client: Existing World Weaver MCP client
            session_id: Optional session ID

        Returns:
            Initialized JarvisMemoryHook
        """
        return cls(ww_client, session_id)

    async def enhance_context(
        self,
        user_text: str,
        session_id: str | None = None,
    ) -> EnhancedContext:
        """
        Get enhanced context for Claude prompt.

        Call this before sending to Claude to inject memory context.

        Args:
            user_text: Transcribed user speech
            session_id: Optional session override

        Returns:
            EnhancedContext with memories and prompt additions
        """
        self._turn_count += 1

        context_parts = []
        memories = []
        personal = None
        proactive = None

        try:
            # Get relevant episodes
            episodes_result = await self.ww.call_tool(
                "mcp__ww-memory__recall_episodes",
                {
                    "query": user_text,
                    "limit": 5,
                    "time_filter": {"after": "7 days ago"},
                }
            )
            memories = episodes_result.get("episodes", [])

            if memories:
                context_parts.append("## Relevant History")
                for ep in memories[:5]:
                    content = ep.get("content", "")[:150]
                    context_parts.append(f"- {content}")

            # Get semantic matches
            entities_result = await self.ww.call_tool(
                "mcp__ww-memory__semantic_recall",
                {
                    "query": user_text,
                    "limit": 5,
                }
            )
            entities = entities_result.get("entities", [])

            if entities:
                context_parts.append("\n## Known Information")
                for ent in entities[:5]:
                    name = ent.get("name", "")
                    summary = ent.get("summary", "")[:80]
                    context_parts.append(f"- **{name}**: {summary}")

            # Get today's calendar for context
            try:
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
                    personal_parts = [f"Today's events ({len(events)}):"]
                    for e in events[:3]:
                        personal_parts.append(f"- {e.get('summary', 'Untitled')}")
                    personal = "\n".join(personal_parts)

                # Check for upcoming event (proactive)
                from datetime import timedelta
                for event in events:
                    start_str = event.get("start", {}).get("dateTime")
                    if start_str:
                        try:
                            start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                            if now <= start <= now + timedelta(minutes=30):
                                mins = int((start - now).total_seconds() / 60)
                                proactive = f"Heads up: {event.get('summary', 'Meeting')} in {mins} minutes."
                                break
                        except ValueError:
                            pass

            except Exception as e:
                logger.debug(f"Could not get calendar context: {e}")

        except Exception as e:
            logger.warning(f"Error enhancing context: {e}")

        return EnhancedContext(
            system_prompt_addition="\n".join(context_parts),
            relevant_memories=memories,
            personal_context=personal,
            proactive_message=proactive,
        )

    async def on_user_speech(
        self,
        text: str,
        session_id: str | None = None,
    ) -> None:
        """
        Called when user speech is transcribed.

        Decides whether to store in memory.

        Args:
            text: Transcribed speech
            session_id: Optional session override
        """
        sid = session_id or self.session_id

        # Import gate for storage decision
        try:
            from t4dm.core.memory_gate import GateContext, MemoryGate, StorageDecision

            gate = MemoryGate(store_threshold=0.4, voice_mode_adjustments=True)
            gate_ctx = GateContext(
                session_id=sid,
                is_voice=True,
            )

            decision = gate.evaluate(text, gate_ctx)

            if decision.decision == StorageDecision.STORE:
                await self._store_episode(
                    f"User: {text}",
                    sid,
                    importance=decision.suggested_importance,
                )

        except Exception as e:
            logger.debug(f"Storage decision error: {e}")

    async def on_response(
        self,
        user_text: str,
        response_text: str,
        session_id: str | None = None,
        was_action: bool = False,
    ) -> None:
        """
        Called after generating response.

        Stores interaction if significant.

        Args:
            user_text: Original user speech
            response_text: Generated response
            session_id: Optional session override
            was_action: Whether an action was performed
        """
        sid = session_id or self.session_id

        # Store if action was taken or contains memory trigger
        memory_triggers = ["i'll remember", "noted", "i've stored", "saving"]
        should_store = was_action or any(t in response_text.lower() for t in memory_triggers)

        if should_store:
            await self._store_episode(
                f"Action: {user_text} -> {response_text[:100]}",
                sid,
                importance=0.6,
            )

    async def on_explicit_memory(
        self,
        content: str,
        session_id: str | None = None,
    ) -> str:
        """
        Store explicit "remember this" request.

        Args:
            content: What to remember
            session_id: Optional session override

        Returns:
            Confirmation message
        """
        sid = session_id or self.session_id

        await self._store_episode(
            f"User asked to remember: {content}",
            sid,
            importance=0.9,
        )

        return "I'll remember that."

    async def on_recall_request(
        self,
        query: str,
        session_id: str | None = None,
    ) -> list[dict]:
        """
        Handle "what do you remember about..." request.

        Args:
            query: What to recall
            session_id: Optional session override

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

    async def start_conversation(self) -> str:
        """
        Called when wake word detected.

        Returns greeting with proactive context.
        """
        self._conversation_id = f"conv-{datetime.now().strftime('%H%M%S')}"
        self._turn_count = 0

        # Get proactive greeting
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        # Check for upcoming events
        try:
            now = datetime.now()
            events_result = await self.ww.call_tool(
                "mcp__google-workspace__calendar_list_events",
                {
                    "date": now.strftime("%Y-%m-%d"),
                    "days": 1,
                    "maxResults": 3,
                }
            )

            events = events_result.get("items", [])
            from datetime import timedelta

            for event in events:
                start_str = event.get("start", {}).get("dateTime")
                if start_str:
                    try:
                        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                        if now <= start <= now + timedelta(hours=1):
                            mins = int((start - now).total_seconds() / 60)
                            return f"{greeting}. Reminder: {event.get('summary', 'meeting')} in {mins} minutes."
                    except ValueError:
                        pass

        except Exception:
            pass

        return f"{greeting}. How can I help?"

    async def end_conversation(self, summary: str | None = None) -> None:
        """
        Called when conversation ends.

        Stores conversation summary if provided.
        """
        if summary:
            await self._store_episode(
                f"Conversation summary: {summary}",
                self.session_id,
                importance=0.5,
            )

        self._conversation_id = None
        self._turn_count = 0

    async def _store_episode(
        self,
        content: str,
        session_id: str,
        importance: float = 0.5,
    ) -> str | None:
        """Store an episode in World Weaver."""
        try:
            result = await self.ww.call_tool(
                "mcp__ww-memory__store_episode",
                {
                    "content": content,
                    "session_id": session_id,
                    "context": {
                        "task": f"voice_conversation:{self._conversation_id or 'default'}",
                    },
                    "emotional_valence": importance,
                }
            )
            return result.get("episode_id")

        except Exception as e:
            logger.warning(f"Failed to store episode: {e}")
            return None

    def close(self) -> None:
        """Clean up resources."""
        self._active = False


# System prompt enhancement for voice mode
VOICE_SYSTEM_PROMPT_ADDITION = """
## Voice Assistant Mode

You are operating as a voice assistant with persistent memory.

VOICE RULES:
1. Be CONCISE - your responses will be spoken aloud. 1-3 sentences ideal.
2. Don't read code - summarize what you did instead.
3. Numbers: say "about fifty" not "47.3"
4. Paths: use filenames only, not full paths.
5. Confirm actions briefly: "Done" not "I have completed..."

MEMORY:
- You have access to past conversations via the context provided.
- When asked "remember that X" - acknowledge and it will be stored.
- When asked "what do you remember about X" - check the context above.
- Reference past interactions naturally when relevant.
"""


def get_enhanced_system_prompt(
    base_prompt: str,
    context: EnhancedContext,
) -> str:
    """
    Build enhanced system prompt with memory context.

    Args:
        base_prompt: Base system prompt
        context: Enhanced context from enhance_context()

    Returns:
        Full system prompt with memory injection
    """
    parts = [base_prompt, VOICE_SYSTEM_PROMPT_ADDITION]

    if context.system_prompt_addition:
        parts.append(f"\n## Memory Context\n{context.system_prompt_addition}")

    if context.personal_context:
        parts.append(f"\n## Current Status\n{context.personal_context}")

    parts.append(f"\nCurrent time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

    return "\n".join(parts)
