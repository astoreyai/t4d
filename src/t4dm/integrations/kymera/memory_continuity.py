"""
Memory Continuity for Kymera Voice.

Provides persistent memory across voice conversations:
- ConversationCapture: Stores complete conversations as threaded episodes
- ProactiveContext: Retrieves relevant past interactions
- ConversationSummarizer: Generates summaries for long-term storage
- MemoryConsolidator: Consolidates related memories during idle time
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from t4dm.integrations.kymera.bridge import VoiceContext

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: str
    timestamp: datetime
    speaker: str  # "user" or "assistant"
    text: str
    intent: str | None = None
    action_taken: str | None = None
    success: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """A complete voice conversation."""
    conversation_id: str
    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    turns: list[ConversationTurn] = field(default_factory=list)
    topic_summary: str | None = None
    actions_taken: list[str] = field(default_factory=list)
    entities_mentioned: list[str] = field(default_factory=list)
    episode_id: UUID | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Conversation duration."""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    @property
    def turn_count(self) -> int:
        """Number of turns."""
        return len(self.turns)

    def to_episode_content(self) -> str:
        """Convert to episode content for storage."""
        parts = [f"Voice conversation ({self.turn_count} turns)"]

        if self.topic_summary:
            parts.append(f"Topic: {self.topic_summary}")

        if self.actions_taken:
            parts.append(f"Actions: {', '.join(self.actions_taken[:5])}")

        # Include key turns
        user_turns = [t for t in self.turns if t.speaker == "user"]
        for turn in user_turns[:5]:
            parts.append(f"User: {turn.text[:100]}")

        return " | ".join(parts)


class ConversationCapture:
    """
    Captures and stores voice conversations.

    Handles:
    - Turn-by-turn capture
    - Conversation threading
    - Summary generation
    - Episode storage
    """

    def __init__(
        self,
        ww_client: Any,
        session_id: str,
        auto_summarize: bool = True,
        summary_threshold_turns: int = 5,
    ):
        """
        Initialize conversation capture.

        Args:
            ww_client: T4DM MCP client
            session_id: Current session ID
            auto_summarize: Generate summaries automatically
            summary_threshold_turns: Turns before generating summary
        """
        self.ww = ww_client
        self.session_id = session_id
        self.auto_summarize = auto_summarize
        self.summary_threshold = summary_threshold_turns

        # Active conversations
        self._active: dict[str, Conversation] = {}
        self._completed: list[Conversation] = []

        logger.info(f"ConversationCapture initialized for session {session_id}")

    def start_conversation(
        self,
        context: VoiceContext | None = None,
    ) -> Conversation:
        """Start capturing a new conversation."""
        conv_id = str(uuid4())

        conversation = Conversation(
            conversation_id=conv_id,
            session_id=context.session_id if context else self.session_id,
            started_at=datetime.now(),
        )

        self._active[conv_id] = conversation
        logger.debug(f"Started conversation {conv_id}")

        return conversation

    def add_turn(
        self,
        conversation_id: str,
        speaker: str,
        text: str,
        intent: str | None = None,
        action: str | None = None,
        success: bool = True,
    ) -> ConversationTurn:
        """Add a turn to the conversation."""
        conversation = self._active.get(conversation_id)
        if not conversation:
            raise ValueError(f"No active conversation: {conversation_id}")

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            timestamp=datetime.now(),
            speaker=speaker,
            text=text,
            intent=intent,
            action_taken=action,
            success=success,
        )

        conversation.turns.append(turn)

        # Track actions
        if action and action not in conversation.actions_taken:
            conversation.actions_taken.append(action)

        # Extract entities from user text
        if speaker == "user":
            entities = self._extract_entities(text)
            for entity in entities:
                if entity not in conversation.entities_mentioned:
                    conversation.entities_mentioned.append(entity)

        return turn

    async def end_conversation(
        self,
        conversation_id: str,
        generate_summary: bool = True,
    ) -> Conversation:
        """
        End and store a conversation.

        Args:
            conversation_id: Conversation to end
            generate_summary: Generate topic summary

        Returns:
            Completed conversation with episode ID
        """
        conversation = self._active.pop(conversation_id, None)
        if not conversation:
            raise ValueError(f"No active conversation: {conversation_id}")

        conversation.ended_at = datetime.now()

        # Generate summary if enabled
        if generate_summary and len(conversation.turns) >= self.summary_threshold:
            conversation.topic_summary = await self._generate_summary(conversation)

        # Store as episode
        episode_id = await self._store_conversation(conversation)
        conversation.episode_id = episode_id

        self._completed.append(conversation)

        logger.info(f"Ended conversation {conversation_id}, episode {episode_id}")
        return conversation

    async def _generate_summary(self, conversation: Conversation) -> str:
        """Generate conversation summary using LLM."""
        # Build transcript
        transcript = []
        for turn in conversation.turns:
            prefix = "User" if turn.speaker == "user" else "Assistant"
            transcript.append(f"{prefix}: {turn.text}")

        transcript_text = "\n".join(transcript)

        # Use Claude to summarize
        try:
            prompt = f"""Summarize this voice conversation in 1-2 sentences. Focus on:
- Main topic discussed
- Actions taken
- Key information shared

Conversation:
{transcript_text}

Summary:"""

            result = await self.ww.call_tool(
                "mcp__anthropic__claude",  # Assumes Claude MCP tool
                {
                    "prompt": prompt,
                    "max_tokens": 100,
                }
            )
            return result.get("text", "Voice conversation")

        except Exception as e:
            logger.debug(f"Summary generation failed: {e}")
            # Fallback to simple summary
            topics = set(conversation.actions_taken[:3])
            return f"Voice conversation about {', '.join(topics) if topics else 'general topics'}"

    async def _store_conversation(self, conversation: Conversation) -> UUID:
        """Store conversation as episode."""
        content = conversation.to_episode_content()

        result = await self.ww.call_tool(
            "mcp__ww-memory__store_episode",
            {
                "content": content,
                "session_id": conversation.session_id,
                "context": {
                    "task": "voice_conversation",
                    "conversation_id": conversation.conversation_id,
                    "turn_count": conversation.turn_count,
                    "actions": conversation.actions_taken,
                },
                "outcome": "success",
                "emotional_valence": 0.5,
            }
        )

        return UUID(result["episode_id"])

    def _extract_entities(self, text: str) -> list[str]:
        """Extract potential entity names from text."""
        import re

        entities = []

        # Capitalized words (potential names)
        words = text.split()
        for word in words:
            clean = re.sub(r"[^\w]", "", word)
            if clean and clean[0].isupper() and len(clean) > 1:
                entities.append(clean)

        # Named patterns
        patterns = [
            r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",  # Full names
            r"(?:@|about|from|to|with) (\w+)",  # Referenced names
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        return list(set(entities))

    def get_active_conversation(self, conversation_id: str) -> Conversation | None:
        """Get active conversation."""
        return self._active.get(conversation_id)

    @property
    def active_count(self) -> int:
        """Number of active conversations."""
        return len(self._active)


class ProactiveContext:
    """
    Provides proactive context from memory.

    Retrieves and ranks relevant past interactions
    to inject into current conversation.
    """

    def __init__(
        self,
        ww_client: Any,
        max_episodes: int = 5,
        max_entities: int = 10,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.7,
    ):
        """
        Initialize proactive context.

        Args:
            ww_client: T4DM MCP client
            max_episodes: Maximum episodes to retrieve
            max_entities: Maximum entities to retrieve
            recency_weight: Weight for recency in scoring
            relevance_weight: Weight for relevance in scoring
        """
        self.ww = ww_client
        self.max_episodes = max_episodes
        self.max_entities = max_entities
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight

        # Context cache
        self._episode_cache: list[dict] = []
        self._entity_cache: list[dict] = []
        self._last_query: str | None = None

    async def get_context_for_query(
        self,
        query: str,
        context: VoiceContext | None = None,
        include_personal: bool = True,
    ) -> dict:
        """
        Get proactive context for a voice query.

        Args:
            query: Current user query
            context: Voice context
            include_personal: Include personal data context

        Returns:
            Dict with episodes, entities, personal, proactive_hints
        """
        result = {
            "episodes": [],
            "entities": [],
            "personal_context": None,
            "proactive_hints": [],
        }

        try:
            # Get relevant episodes
            episodes_result = await self.ww.call_tool(
                "mcp__ww-memory__recall_episodes",
                {
                    "query": query,
                    "limit": self.max_episodes,
                    "time_filter": {"after": "30 days ago"},
                }
            )
            result["episodes"] = self._rank_episodes(
                episodes_result.get("episodes", [])
            )

            # Get relevant entities
            entities_result = await self.ww.call_tool(
                "mcp__ww-memory__semantic_recall",
                {
                    "query": query,
                    "limit": self.max_entities,
                }
            )
            result["entities"] = entities_result.get("entities", [])

            # Generate proactive hints
            result["proactive_hints"] = await self._generate_hints(
                query, result["episodes"], result["entities"]
            )

            # Personal context
            if include_personal:
                result["personal_context"] = await self._get_personal_context()

            # Cache for follow-up queries
            self._episode_cache = result["episodes"]
            self._entity_cache = result["entities"]
            self._last_query = query

        except Exception as e:
            logger.error(f"Failed to get proactive context: {e}")

        return result

    def _rank_episodes(self, episodes: list[dict]) -> list[dict]:
        """Rank episodes by recency and relevance."""
        now = datetime.now()

        def score_episode(ep: dict) -> float:
            # Recency score (decay over 30 days)
            timestamp = ep.get("timestamp")
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp)
                    days_ago = (now - ts).days
                    recency = max(0, 1 - days_ago / 30)
                except ValueError:
                    recency = 0.5
            else:
                recency = 0.5

            # Relevance score (from retrieval)
            relevance = ep.get("score", 0.5)

            # Combined score
            return (
                self.recency_weight * recency +
                self.relevance_weight * relevance
            )

        ranked = sorted(episodes, key=score_episode, reverse=True)
        return ranked[:self.max_episodes]

    async def _generate_hints(
        self,
        query: str,
        episodes: list[dict],
        entities: list[dict],
    ) -> list[str]:
        """Generate proactive hints from context."""
        hints = []

        # Check for related past conversations
        for ep in episodes[:3]:
            content = ep.get("content", "")
            if "voice conversation" in content.lower():
                hints.append("Related past conversation available")
                break

        # Check for known entities
        query_lower = query.lower()
        for ent in entities[:5]:
            name = ent.get("name", "").lower()
            if name in query_lower:
                summary = ent.get("summary", "")[:50]
                hints.append(f"Known: {name} - {summary}")

        return hints[:3]

    async def _get_personal_context(self) -> str | None:
        """Get personal context summary."""
        try:
            now = datetime.now()

            # Get today's events
            events_result = await self.ww.call_tool(
                "mcp__google-workspace__calendar_list_events",
                {
                    "date": now.strftime("%Y-%m-%d"),
                    "days": 1,
                    "maxResults": 3,
                }
            )

            events = events_result.get("items", [])
            if events:
                parts = [f"Today: {len(events)} events"]
                for e in events[:2]:
                    parts.append(f"- {e.get('summary', 'Event')}")
                return ". ".join(parts)

        except Exception:
            pass

        return None

    def get_follow_up_context(self) -> dict:
        """Get context for follow-up queries (uses cache)."""
        return {
            "episodes": self._episode_cache,
            "entities": self._entity_cache,
            "previous_query": self._last_query,
        }


class ConversationSummarizer:
    """
    Generates summaries of conversations for long-term memory.

    Extracts:
    - Key topics discussed
    - Actions taken
    - Entities mentioned
    - User preferences/patterns
    """

    def __init__(self, ww_client: Any):
        self.ww = ww_client

    async def summarize_daily_conversations(
        self,
        session_id: str,
        date: datetime | None = None,
    ) -> str:
        """
        Summarize all conversations from a day.

        Args:
            session_id: Session to summarize
            date: Date to summarize (default: today)

        Returns:
            Daily summary
        """
        target_date = date or datetime.now()
        start = target_date.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)

        # Get day's episodes
        result = await self.ww.call_tool(
            "mcp__ww-memory__recall_episodes",
            {
                "query": "voice conversation",
                "limit": 50,
                "time_filter": {
                    "after": start.isoformat(),
                    "before": end.isoformat(),
                },
            }
        )

        episodes = result.get("episodes", [])
        if not episodes:
            return f"No voice conversations on {target_date.strftime('%B %d')}"

        # Extract topics and actions
        topics = []
        actions = []

        for ep in episodes:
            content = ep.get("content", "")
            context = ep.get("context", {})

            # Extract topic
            if "Topic:" in content:
                topic = content.split("Topic:")[1].split("|")[0].strip()
                if topic not in topics:
                    topics.append(topic)

            # Extract actions
            ep_actions = context.get("actions", [])
            for action in ep_actions:
                if action not in actions:
                    actions.append(action)

        # Build summary
        parts = [f"Daily summary for {target_date.strftime('%B %d')}:"]
        parts.append(f"- {len(episodes)} conversations")

        if topics:
            parts.append(f"- Topics: {', '.join(topics[:5])}")

        if actions:
            parts.append(f"- Actions: {', '.join(actions[:5])}")

        return "\n".join(parts)


class MemoryConsolidator:
    """
    Consolidates related memories during idle time.

    Performs:
    - Duplicate detection and merging
    - Entity extraction from episodes
    - Relationship inference
    - Skill pattern detection
    """

    def __init__(self, ww_client: Any):
        self.ww = ww_client

    async def consolidate_recent(
        self,
        hours: int = 24,
        session_id: str | None = None,
    ) -> dict:
        """
        Consolidate recent memories.

        Args:
            hours: How far back to consolidate
            session_id: Optional session filter

        Returns:
            Consolidation stats
        """
        stats = {
            "episodes_processed": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "duplicates_merged": 0,
        }

        try:
            # Get recent episodes
            result = await self.ww.call_tool(
                "mcp__ww-memory__recall_episodes",
                {
                    "query": "",
                    "limit": 100,
                    "time_filter": {"after": f"{hours} hours ago"},
                }
            )

            episodes = result.get("episodes", [])
            stats["episodes_processed"] = len(episodes)

            # Extract entities from episodes
            for ep in episodes:
                content = ep.get("content", "")
                entities = self._extract_entities_from_episode(content)

                for entity in entities:
                    created = await self._ensure_entity_exists(entity)
                    if created:
                        stats["entities_created"] += 1

            # Detect patterns for skills
            patterns = self._detect_patterns(episodes)
            for pattern in patterns:
                await self._create_or_update_skill(pattern)

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

        return stats

    def _extract_entities_from_episode(self, content: str) -> list[dict]:
        """Extract potential entities from episode content."""
        import re

        entities = []

        # Look for action patterns that mention names
        action_patterns = [
            r"email(?:ed)?\s+(?:to\s+)?(\w+)",
            r"call(?:ed)?\s+(\w+)",
            r"meeting\s+with\s+(\w+)",
            r"reminded?\s+(?:about\s+)?(\w+)",
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match[0].isupper():  # Likely a name
                    entities.append({
                        "name": match,
                        "type": "contact",
                        "source": "voice_conversation",
                    })

        return entities

    async def _ensure_entity_exists(self, entity: dict) -> bool:
        """Ensure entity exists in knowledge graph."""
        try:
            # Check if exists
            result = await self.ww.call_tool(
                "mcp__ww-memory__semantic_recall",
                {"query": entity["name"], "limit": 1}
            )

            existing = result.get("entities", [])
            if existing:
                return False  # Already exists

            # Create new entity
            await self.ww.call_tool(
                "mcp__ww-memory__create_entity",
                {
                    "name": entity["name"],
                    "type": entity.get("type", "unknown"),
                    "summary": "Mentioned in voice conversation",
                    "source": entity.get("source", "voice"),
                }
            )
            return True

        except Exception as e:
            logger.debug(f"Entity creation failed: {e}")
            return False

    def _detect_patterns(self, episodes: list[dict]) -> list[dict]:
        """Detect repeating patterns that could become skills."""
        # Group by action type
        action_counts: dict[str, int] = {}

        for ep in episodes:
            context = ep.get("context", {})
            actions = context.get("actions", [])

            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1

        # Patterns are actions repeated 3+ times
        patterns = []
        for action, count in action_counts.items():
            if count >= 3:
                patterns.append({
                    "action": action,
                    "count": count,
                    "type": "frequent_action",
                })

        return patterns

    async def _create_or_update_skill(self, pattern: dict) -> None:
        """Create or update skill from pattern."""
        # This would create a Procedure in WW
        # For now, just log
        logger.info(f"Detected pattern: {pattern}")
