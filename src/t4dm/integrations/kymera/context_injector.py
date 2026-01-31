"""
Context Injector - Injects World Weaver memory into Claude prompts.

Enhances voice assistant responses with relevant context from:
- Past conversations
- Learned knowledge
- Personal data
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from t4dm.integrations.kymera.bridge import MemoryContext, VoiceContext, VoiceMemoryBridge
from t4dm.learning.events import ToonJSON

logger = logging.getLogger(__name__)


@dataclass
class InjectionConfig:
    """Configuration for context injection."""
    max_episodes: int = 5
    max_entities: int = 10
    max_skills: int = 3
    include_timestamps: bool = True
    include_personal: bool = True
    max_context_chars: int = 2000
    use_toon_json: bool = True  # E1: Token-optimized format (~50% reduction)


class ContextInjector:
    """
    Injects World Weaver memory context into Claude prompts.

    Modifies the system prompt or adds context messages to include
    relevant memories, knowledge, and personal data.
    """

    VOICE_SYSTEM_PROMPT = """You are Kymera, a voice assistant with persistent memory.

CRITICAL VOICE RULES:
1. Be CONCISE - responses will be spoken aloud. Aim for 1-3 sentences.
2. Don't read code aloud - summarize what you did.
3. Use natural, conversational language.
4. Confirm actions briefly: "Done" not "I have completed the task..."
5. Numbers: say "about fifty" not "approximately 47.3"
6. Paths: just the filename, not full paths.
7. You can interrupt at any time - no need to say everything.

MEMORY CAPABILITIES:
- You have access to past conversations and learned knowledge.
- When asked "what do you remember about X?" - actually check your memory context.
- When told "remember that X" - acknowledge and it will be stored.
- Reference past interactions naturally: "As we discussed yesterday..."

PERSONAL DATA ACCESS:
- You can access calendar, email, and contacts.
- For time-sensitive queries, check the current context below.
- Summarize appointments and emails conversationally.
"""

    # ToonJSON legend for context injection (helps LLM interpret compact format)
    TOON_LEGEND = """[Legend: eps=history, ents=known, sk=skills, pc=status, c=content, n=name, sum=summary, d=description, ts=timestamp]"""

    def __init__(
        self,
        memory_bridge: VoiceMemoryBridge,
        config: InjectionConfig | None = None,
    ):
        """
        Initialize context injector.

        Args:
            memory_bridge: Voice memory bridge for retrieving context
            config: Injection configuration
        """
        self.bridge = memory_bridge
        self.config = config or InjectionConfig()
        self._toon_json = ToonJSON()  # E1: Token-optimized encoder

    async def build_system_prompt(
        self,
        query: str,
        context: VoiceContext,
        base_prompt: str | None = None,
    ) -> str:
        """
        Build system prompt with injected context.

        Args:
            query: Current user query
            context: Voice interaction context
            base_prompt: Optional base prompt to extend

        Returns:
            Enhanced system prompt with memory context
        """
        prompt_parts = [base_prompt or self.VOICE_SYSTEM_PROMPT]

        # Get relevant memory context
        memory_ctx = await self.bridge.get_relevant_context(
            query, context,
            include_personal=self.config.include_personal,
        )

        # Format and inject context
        context_section = self._format_context(memory_ctx)
        if context_section:
            prompt_parts.append(f"\n{context_section}")

        # Add current time context
        prompt_parts.append(f"\nCurrent time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

        return "\n".join(prompt_parts)

    async def build_context_message(
        self,
        query: str,
        context: VoiceContext,
    ) -> str | None:
        """
        Build a context message to prepend to conversation.

        Alternative to system prompt injection - adds a system message
        with context before the user's query.

        Args:
            query: Current user query
            context: Voice interaction context

        Returns:
            Context message or None if no relevant context
        """
        memory_ctx = await self.bridge.get_relevant_context(
            query, context,
            include_personal=self.config.include_personal,
        )

        formatted = self._format_context(memory_ctx)
        if not formatted:
            return None

        return f"[Memory Context]\n{formatted}"

    def _format_context(self, memory_ctx: MemoryContext) -> str:
        """Format memory context for prompt injection."""
        # E1: Use ToonJSON if enabled for ~50% token reduction
        if self.config.use_toon_json:
            return self._format_context_toon(memory_ctx)
        return self._format_context_verbose(memory_ctx)

    def _format_context_toon(self, memory_ctx: MemoryContext) -> str:
        """E1: Format memory context using ToonJSON for token efficiency."""
        context_dict = {}

        # Recent relevant episodes
        if memory_ctx.episodes:
            eps = []
            for ep in memory_ctx.episodes[:self.config.max_episodes]:
                ep_data = {"content": ep.get("content", "")[:150]}
                if self.config.include_timestamps and ep.get("timestamp"):
                    ep_data["timestamp"] = ep.get("timestamp")
                eps.append(ep_data)
            if eps:
                context_dict["episodes"] = eps

        # Key entities/knowledge
        if memory_ctx.entities:
            ents = []
            for ent in memory_ctx.entities[:self.config.max_entities]:
                ents.append({
                    "name": ent.get("name", ""),
                    "summary": ent.get("summary", "")[:80],
                })
            if ents:
                context_dict["entities"] = ents

        # Applicable skills
        if memory_ctx.skills:
            sk = []
            for skill in memory_ctx.skills[:self.config.max_skills]:
                sk.append({
                    "name": skill.get("name", ""),
                    "description": skill.get("description", "")[:60],
                })
            if sk:
                context_dict["skills"] = sk

        # Personal context
        if memory_ctx.personal_context:
            context_dict["personal_context"] = memory_ctx.personal_context

        if not context_dict:
            return ""

        # Encode with ToonJSON and prepend legend
        encoded = self._toon_json.encode(context_dict)
        return f"{self.TOON_LEGEND}\n{encoded}"

    def _format_context_verbose(self, memory_ctx: MemoryContext) -> str:
        """Format memory context in verbose markdown (original format)."""
        sections = []
        total_chars = 0

        # Recent relevant episodes
        if memory_ctx.episodes:
            episode_lines = ["## Relevant History"]
            for ep in memory_ctx.episodes[:self.config.max_episodes]:
                content = ep.get("content", "")[:150]
                if self.config.include_timestamps:
                    ts = ep.get("timestamp", "")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            ts_str = dt.strftime("%b %d")
                            content = f"[{ts_str}] {content}"
                        except ValueError:
                            pass
                episode_lines.append(f"- {content}")

            section = "\n".join(episode_lines)
            if total_chars + len(section) < self.config.max_context_chars:
                sections.append(section)
                total_chars += len(section)

        # Key entities/knowledge
        if memory_ctx.entities:
            entity_lines = ["## Known Information"]
            for ent in memory_ctx.entities[:self.config.max_entities]:
                name = ent.get("name", "Unknown")
                summary = ent.get("summary", "")[:80]
                entity_lines.append(f"- **{name}**: {summary}")

            section = "\n".join(entity_lines)
            if total_chars + len(section) < self.config.max_context_chars:
                sections.append(section)
                total_chars += len(section)

        # Applicable skills
        if memory_ctx.skills:
            skill_lines = ["## You Know How To"]
            for skill in memory_ctx.skills[:self.config.max_skills]:
                name = skill.get("name", "Unknown")
                desc = skill.get("description", "")[:60]
                skill_lines.append(f"- {name}: {desc}")

            section = "\n".join(skill_lines)
            if total_chars + len(section) < self.config.max_context_chars:
                sections.append(section)
                total_chars += len(section)

        # Personal context
        if memory_ctx.personal_context:
            personal_section = f"## Current Status\n{memory_ctx.personal_context}"
            if total_chars + len(personal_section) < self.config.max_context_chars:
                sections.append(personal_section)

        return "\n\n".join(sections)

    async def get_proactive_context(self, context: VoiceContext) -> str | None:
        """
        Get proactive context to mention at conversation start.

        Returns things like:
        - Upcoming calendar events
        - Overdue tasks
        - Important unread emails

        Args:
            context: Voice interaction context

        Returns:
            Proactive context string or None
        """
        proactive_parts = []

        try:
            # Check for upcoming events in next hour
            from datetime import timedelta

            now = datetime.now()
            events_result = await self.bridge.ww.call_tool(
                "mcp__google-workspace__calendar_list_events",
                {
                    "date": now.strftime("%Y-%m-%d"),
                    "days": 1,
                    "maxResults": 10,
                }
            )

            events = events_result.get("items", [])
            upcoming = []
            for event in events:
                start_str = event.get("start", {}).get("dateTime")
                if start_str:
                    try:
                        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                        if now <= start <= now + timedelta(hours=2):
                            upcoming.append(event)
                    except ValueError:
                        pass

            if upcoming:
                next_event = upcoming[0]
                start_str = next_event.get("start", {}).get("dateTime", "")
                try:
                    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    mins = int((start - now).total_seconds() / 60)
                    if mins <= 30:
                        proactive_parts.append(
                            f"Reminder: {next_event.get('summary', 'Meeting')} "
                            f"starts in {mins} minutes."
                        )
                except ValueError:
                    pass

        except Exception as e:
            logger.debug(f"Could not get proactive context: {e}")

        return " ".join(proactive_parts) if proactive_parts else None


class ConversationContextManager:
    """
    Manages context across a voice conversation.

    Tracks what's been discussed to avoid repetition
    and maintain coherent context.
    """

    def __init__(self, injector: ContextInjector):
        self.injector = injector
        self._mentioned_episodes: set[str] = set()
        self._mentioned_entities: set[str] = set()
        self._conversation_topics: list[str] = []

    def track_context_used(self, memory_ctx: MemoryContext) -> None:
        """Track which context items have been used."""
        for ep in memory_ctx.episodes:
            ep_id = ep.get("id")
            if ep_id:
                self._mentioned_episodes.add(ep_id)

        for ent in memory_ctx.entities:
            ent_name = ent.get("name")
            if ent_name:
                self._mentioned_entities.add(ent_name)

    def add_topic(self, topic: str) -> None:
        """Add a discussed topic."""
        self._conversation_topics.append(topic)

    def filter_redundant(self, memory_ctx: MemoryContext) -> MemoryContext:
        """Filter out already-mentioned context."""
        filtered_episodes = [
            ep for ep in memory_ctx.episodes
            if ep.get("id") not in self._mentioned_episodes
        ]

        filtered_entities = [
            ent for ent in memory_ctx.entities
            if ent.get("name") not in self._mentioned_entities
        ]

        return MemoryContext(
            episodes=filtered_episodes,
            entities=filtered_entities,
            skills=memory_ctx.skills,
            personal_context=memory_ctx.personal_context,
        )

    def reset(self) -> None:
        """Reset for new conversation."""
        self._mentioned_episodes.clear()
        self._mentioned_entities.clear()
        self._conversation_topics.clear()
