"""
Voice Intent Parser - Maps voice commands to structured actions.

Uses pattern matching for common commands and LLM fallback for complex intents.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ww.core.actions import ActionCategory, ActionRequest, ActionStatus

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Result of intent parsing."""
    action_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_text: str = ""
    extracted_entities: list[str] = field(default_factory=list)


class VoiceIntentParser:
    """
    Parses voice input into structured action intents.

    Two-stage parsing:
    1. Fast pattern matching for common commands
    2. LLM fallback for complex/ambiguous intents
    """

    # Pattern groups for intent matching
    # Format: (pattern, action_name, parameter_extractor)

    EMAIL_PATTERNS = [
        (r"(?:send|write|compose)\s+(?:an?\s+)?email\s+to\s+(.+?)(?:\s+(?:about|saying|with subject)\s+(.+))?$",
         "email.send", ["recipient", "subject"]),
        (r"(?:read|check|show)\s+(?:my\s+)?(?:latest|recent|new|unread)\s+emails?",
         "email.read", []),
        (r"(?:read|check|show)\s+(?:my\s+)?emails?\s+(?:from|by)\s+(.+)",
         "email.search", ["from"]),
        (r"reply\s+to\s+(.+?)(?:\s+(?:saying|with)\s+(.+))?$",
         "email.reply", ["to", "content"]),
        (r"forward\s+(?:that\s+)?(?:email\s+)?to\s+(.+)",
         "email.forward", ["to"]),
    ]

    CALENDAR_PATTERNS = [
        (r"(?:schedule|create|add|set up)\s+(?:a\s+)?(?:meeting|event|appointment)\s+(?:with\s+)?(.+?)(?:\s+(?:at|on|for)\s+(.+))?$",
         "calendar.create", ["title", "when"]),
        (r"what(?:'s| is)\s+(?:on\s+)?(?:my\s+)?(?:calendar|schedule)(?:\s+(?:today|tomorrow|this week))?",
         "calendar.list", ["when"]),
        (r"when\s+(?:is|am)\s+(?:my\s+)?(?:next\s+)?(.+)",
         "calendar.query", ["query"]),
        (r"(?:cancel|delete|remove)\s+(?:my\s+)?(?:meeting|event|appointment)\s+(?:with\s+)?(.+)",
         "calendar.delete", ["title"]),
        (r"(?:reschedule|move)\s+(?:my\s+)?(.+?)\s+to\s+(.+)",
         "calendar.update", ["title", "new_time"]),
    ]

    REMINDER_PATTERNS = [
        (r"remind\s+me\s+(?:to\s+)?(.+?)(?:\s+(?:at|in|on|tomorrow|tonight)\s*(.*))?$",
         "reminder.set", ["message", "when"]),
        (r"set\s+(?:a\s+)?(?:reminder|alarm)\s+(?:for\s+)?(.+)",
         "reminder.set", ["when"]),
        (r"(?:what|show)\s+(?:are\s+)?(?:my\s+)?reminders?",
         "reminder.list", []),
        (r"(?:cancel|delete|remove)\s+(?:the\s+)?reminder\s+(?:about|for|to)\s+(.+)",
         "reminder.cancel", ["message"]),
    ]

    TASK_PATTERNS = [
        (r"add\s+(?:a\s+)?(?:task|todo|to-do)\s*(?:to\s+)?(.+)",
         "task.create", ["title"]),
        (r"(?:what|show)\s+(?:are\s+)?(?:my\s+)?(?:tasks?|todos?|to-dos?)",
         "task.list", []),
        (r"(?:mark|set)\s+(.+?)\s+(?:as\s+)?(?:done|complete|finished)",
         "task.complete", ["title"]),
        (r"(?:i|we)\s+(?:finished|completed|done with)\s+(.+)",
         "task.complete", ["title"]),
    ]

    MEMORY_PATTERNS = [
        (r"remember\s+(?:that\s+)?(.+)",
         "memory.store", ["content"]),
        (r"(?:don't|do not)\s+forget\s+(?:that\s+)?(.+)",
         "memory.store", ["content"]),
        (r"(?:what\s+)?(?:do\s+)?you\s+(?:know|remember)\s+about\s+(.+)",
         "memory.recall", ["query"]),
        (r"forget\s+(?:about\s+)?(.+)",
         "memory.forget", ["query"]),
        (r"(?:who|what)\s+(?:is|was)\s+(.+)",
         "memory.recall", ["query"]),
    ]

    CONTACT_PATTERNS = [
        (r"(?:who\s+is|tell me about)\s+(.+)",
         "contact.lookup", ["name"]),
        (r"(?:what's|when is)\s+(.+?)(?:'s)?\s+birthday",
         "contact.birthday", ["name"]),
        (r"(?:call|phone)\s+(.+)",
         "call.initiate", ["contact"]),
    ]

    SYSTEM_PATTERNS = [
        (r"(?:set\s+)?(?:a\s+)?timer\s+(?:for\s+)?(.+)",
         "timer.set", ["duration"]),
        (r"(?:what|what's)\s+(?:the\s+)?time",
         "lookup.time", []),
        (r"(?:what|what's)\s+(?:the\s+)?(?:date|day)",
         "lookup.date", []),
        (r"(?:what's|how's)\s+the\s+weather",
         "lookup.weather", []),
        (r"(?:search|google|look up)\s+(.+)",
         "search.web", ["query"]),
    ]

    FILE_PATTERNS = [
        (r"(?:open|show|read)\s+(?:the\s+)?file\s+(.+)",
         "file.read", ["filename"]),
        (r"(?:create|make|new)\s+(?:a\s+)?file\s+(?:called\s+)?(.+)",
         "file.create", ["filename"]),
        (r"(?:search|find)\s+(?:files?\s+)?(?:for\s+)?(.+)",
         "search.files", ["query"]),
    ]

    def __init__(self, llm_client: Any | None = None):
        """
        Initialize intent parser.

        Args:
            llm_client: Optional LLM client for complex intent parsing
        """
        self.llm = llm_client

        # Compile all patterns
        self._patterns: list[tuple[re.Pattern, str, list[str]]] = []
        pattern_groups = [
            self.EMAIL_PATTERNS,
            self.CALENDAR_PATTERNS,
            self.REMINDER_PATTERNS,
            self.TASK_PATTERNS,
            self.MEMORY_PATTERNS,
            self.CONTACT_PATTERNS,
            self.SYSTEM_PATTERNS,
            self.FILE_PATTERNS,
        ]

        for group in pattern_groups:
            for pattern, action, params in group:
                self._patterns.append((
                    re.compile(pattern, re.IGNORECASE),
                    action,
                    params,
                ))

        logger.info(f"VoiceIntentParser initialized with {len(self._patterns)} patterns")

    def parse(self, text: str) -> ParsedIntent:
        """
        Parse voice input to action intent.

        Args:
            text: Transcribed voice input

        Returns:
            ParsedIntent with action and parameters
        """
        text = text.strip()

        # Try pattern matching first
        for pattern, action, param_names in self._patterns:
            match = pattern.match(text)
            if match:
                # Extract parameters from groups
                params = {}
                for i, name in enumerate(param_names):
                    if i < len(match.groups()) and match.group(i + 1):
                        params[name] = match.group(i + 1).strip()

                return ParsedIntent(
                    action_name=action,
                    parameters=params,
                    confidence=0.9,
                    raw_text=text,
                )

        # No pattern match - fall through to chat
        return ParsedIntent(
            action_name="claude.chat",
            parameters={"message": text},
            confidence=0.5,
            raw_text=text,
        )

    async def parse_with_llm(self, text: str) -> ParsedIntent:
        """
        Parse using LLM for complex intents.

        Falls back to pattern matching if LLM unavailable.

        Args:
            text: Transcribed voice input

        Returns:
            ParsedIntent with action and parameters
        """
        # First try patterns
        pattern_result = self.parse(text)
        if pattern_result.confidence >= 0.9:
            return pattern_result

        # Use LLM if available and pattern didn't match well
        if self.llm and pattern_result.action_name == "claude.chat":
            try:
                llm_result = await self._llm_parse(text)
                if llm_result.confidence > pattern_result.confidence:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}")

        return pattern_result

    async def _llm_parse(self, text: str) -> ParsedIntent:
        """Use LLM to parse complex intent."""
        available_actions = [
            "email.send", "email.read", "email.reply", "email.forward",
            "calendar.create", "calendar.list", "calendar.query", "calendar.delete",
            "reminder.set", "reminder.list",
            "task.create", "task.list", "task.complete",
            "memory.store", "memory.recall", "memory.forget",
            "contact.lookup", "call.initiate",
            "search.web", "search.files",
            "file.read", "file.create",
            "claude.chat",  # Default fallback
        ]

        prompt = f"""Parse this voice command into a structured action.

Voice command: "{text}"

Available actions: {available_actions}

Return a JSON object:
{{
    "action": "action.name",
    "parameters": {{"key": "value"}},
    "confidence": 0.0 to 1.0
}}

If the command is a general question or doesn't fit any action, use "claude.chat" with the message as parameter.

Only return the JSON, nothing else."""

        # Call LLM (this would use httpx or similar)
        response = await self.llm.complete(prompt, max_tokens=200)

        # Parse JSON response
        import json
        try:
            result = json.loads(response)
            return ParsedIntent(
                action_name=result.get("action", "claude.chat"),
                parameters=result.get("parameters", {"message": text}),
                confidence=result.get("confidence", 0.7),
                raw_text=text,
            )
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM response: {response}")
            return ParsedIntent(
                action_name="claude.chat",
                parameters={"message": text},
                confidence=0.5,
                raw_text=text,
            )

    def to_action_request(
        self,
        intent: ParsedIntent,
        session_id: str,
    ) -> ActionRequest:
        """
        Convert parsed intent to ActionRequest.

        Args:
            intent: Parsed intent
            session_id: Current session ID

        Returns:
            ActionRequest ready for routing
        """
        # Map action name to category
        category_map = {
            "email": ActionCategory.EMAIL,
            "calendar": ActionCategory.CALENDAR,
            "reminder": ActionCategory.REMINDER,
            "task": ActionCategory.TASK,
            "memory": ActionCategory.SYSTEM,
            "contact": ActionCategory.LOOKUP,
            "call": ActionCategory.CALL,
            "search": ActionCategory.SEARCH,
            "file": ActionCategory.FILE,
            "timer": ActionCategory.TIMER,
            "lookup": ActionCategory.LOOKUP,
            "claude": ActionCategory.SYSTEM,
        }

        action_prefix = intent.action_name.split(".")[0]
        category = category_map.get(action_prefix, ActionCategory.SYSTEM)

        return ActionRequest(
            id=uuid4(),
            action_name=intent.action_name,
            category=category,
            parameters=intent.parameters,
            session_id=session_id,
            user_utterance=intent.raw_text,
            extracted_intent=intent.action_name,
            status=ActionStatus.PENDING,
        )


class TimeParser:
    """Parse natural language time expressions."""

    RELATIVE_PATTERNS = [
        (r"in\s+(\d+)\s+minutes?", "minutes"),
        (r"in\s+(\d+)\s+hours?", "hours"),
        (r"in\s+(\d+)\s+days?", "days"),
        (r"tomorrow\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", "tomorrow"),
        (r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", "time"),
        (r"tomorrow", "tomorrow_default"),
        (r"tonight", "tonight"),
        (r"this\s+(?:afternoon|evening)", "later_today"),
    ]

    def __init__(self):
        self._patterns = [
            (re.compile(p, re.IGNORECASE), t)
            for p, t in self.RELATIVE_PATTERNS
        ]

    def parse(self, text: str) -> datetime | None:
        """
        Parse time expression to datetime.

        Args:
            text: Natural language time expression

        Returns:
            Parsed datetime or None
        """
        now = datetime.now()

        for pattern, time_type in self._patterns:
            match = pattern.search(text)
            if match:
                if time_type == "minutes":
                    mins = int(match.group(1))
                    return now + timedelta(minutes=mins)

                if time_type == "hours":
                    hours = int(match.group(1))
                    return now + timedelta(hours=hours)

                if time_type == "days":
                    days = int(match.group(1))
                    return now + timedelta(days=days)

                if time_type == "tomorrow":
                    hour = int(match.group(1))
                    minute = int(match.group(2) or 0)
                    ampm = match.group(3)
                    if ampm and ampm.lower() == "pm" and hour < 12:
                        hour += 12
                    elif ampm and ampm.lower() == "am" and hour == 12:
                        hour = 0
                    tomorrow = now + timedelta(days=1)
                    return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)

                if time_type == "time":
                    hour = int(match.group(1))
                    minute = int(match.group(2) or 0)
                    ampm = match.group(3)
                    if ampm and ampm.lower() == "pm" and hour < 12:
                        hour += 12
                    elif ampm and ampm.lower() == "am" and hour == 12:
                        hour = 0
                    result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    # If time is in past, assume tomorrow
                    if result < now:
                        result += timedelta(days=1)
                    return result

                if time_type == "tomorrow_default":
                    tomorrow = now + timedelta(days=1)
                    return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)

                if time_type == "tonight":
                    return now.replace(hour=20, minute=0, second=0, microsecond=0)

                if time_type == "later_today":
                    return now.replace(hour=17, minute=0, second=0, microsecond=0)

        return None
