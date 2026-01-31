"""
Advanced Features for Kymera Voice + World Weaver.

Phase 5 capabilities:
- NotificationManager: Push notifications for important events
- PreferenceLearner: Learn and apply user patterns
- MultiModalContext: Handle images/files in voice context
- VoiceTriggers: Custom wake word handlers
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Notifications
# =============================================================================

class NotificationPriority(str, Enum):
    """Notification priority levels."""
    CRITICAL = "critical"   # Interrupt immediately
    HIGH = "high"           # Announce at next opportunity
    NORMAL = "normal"       # Queue for batch
    LOW = "low"             # Silent/log only


class NotificationType(str, Enum):
    """Types of notifications."""
    CALENDAR = "calendar"
    EMAIL = "email"
    TASK = "task"
    REMINDER = "reminder"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class Notification:
    """A notification to deliver."""
    id: str
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    created_at: datetime = field(default_factory=datetime.now)
    deliver_at: datetime | None = None
    spoken_text: str | None = None  # Override for voice
    action_url: str | None = None
    metadata: dict = field(default_factory=dict)
    delivered: bool = False
    acknowledged: bool = False


class NotificationManager:
    """
    Manages notifications for voice assistant.

    Handles:
    - Queuing notifications by priority
    - Scheduling delivery
    - Voice announcement formatting
    - Quiet hours
    """

    def __init__(
        self,
        tts_callback: Callable[[str], Awaitable[None]] | None = None,
        quiet_hours_start: time | None = None,
        quiet_hours_end: time | None = None,
    ):
        """
        Initialize notification manager.

        Args:
            tts_callback: Async callback to speak notifications
            quiet_hours_start: Start of quiet hours (e.g., 22:00)
            quiet_hours_end: End of quiet hours (e.g., 07:00)
        """
        self.tts = tts_callback
        self.quiet_start = quiet_hours_start or time(22, 0)
        self.quiet_end = quiet_hours_end or time(7, 0)

        # Notification queues by priority
        self._critical_queue: list[Notification] = []
        self._high_queue: list[Notification] = []
        self._normal_queue: list[Notification] = []
        self._low_queue: list[Notification] = []

        # Scheduled notifications
        self._scheduled: dict[str, Notification] = {}

        # Delivery history
        self._delivered: list[Notification] = []

        logger.info("NotificationManager initialized")

    def queue(self, notification: Notification) -> None:
        """Add notification to appropriate queue."""
        if notification.deliver_at and notification.deliver_at > datetime.now():
            # Schedule for later
            self._scheduled[notification.id] = notification
            logger.debug(f"Scheduled notification {notification.id} for {notification.deliver_at}")
            return

        # Add to priority queue
        if notification.priority == NotificationPriority.CRITICAL:
            self._critical_queue.append(notification)
        elif notification.priority == NotificationPriority.HIGH:
            self._high_queue.append(notification)
        elif notification.priority == NotificationPriority.NORMAL:
            self._normal_queue.append(notification)
        else:
            self._low_queue.append(notification)

        logger.debug(f"Queued {notification.priority.value} notification: {notification.title}")

    def create_calendar_notification(
        self,
        event_title: str,
        event_start: datetime,
        minutes_before: int = 15,
    ) -> Notification:
        """Create a calendar event notification."""
        deliver_at = event_start - timedelta(minutes=minutes_before)

        return Notification(
            id=str(uuid4()),
            type=NotificationType.CALENDAR,
            priority=NotificationPriority.HIGH,
            title=f"Upcoming: {event_title}",
            message=f"{event_title} starts in {minutes_before} minutes",
            spoken_text=f"Heads up. {event_title} starts in {minutes_before} minutes.",
            deliver_at=deliver_at,
            metadata={"event_start": event_start.isoformat()},
        )

    def create_email_notification(
        self,
        sender: str,
        subject: str,
        is_important: bool = False,
    ) -> Notification:
        """Create an email notification."""
        priority = NotificationPriority.HIGH if is_important else NotificationPriority.NORMAL

        return Notification(
            id=str(uuid4()),
            type=NotificationType.EMAIL,
            priority=priority,
            title=f"Email from {sender}",
            message=subject,
            spoken_text=f"New email from {sender}. Subject: {subject}",
        )

    def create_reminder_notification(
        self,
        message: str,
        deliver_at: datetime,
    ) -> Notification:
        """Create a reminder notification."""
        return Notification(
            id=str(uuid4()),
            type=NotificationType.REMINDER,
            priority=NotificationPriority.HIGH,
            title="Reminder",
            message=message,
            spoken_text=f"Reminder: {message}",
            deliver_at=deliver_at,
        )

    async def get_pending(self) -> list[Notification]:
        """
        Get pending notifications to deliver.

        Respects quiet hours and priority ordering.
        """
        # Check scheduled notifications
        now = datetime.now()
        due_notifications = []

        for notif_id, notif in list(self._scheduled.items()):
            if notif.deliver_at and notif.deliver_at <= now:
                due_notifications.append(notif)
                del self._scheduled[notif_id]
                self.queue(notif)

        # Check quiet hours
        if self._is_quiet_hours():
            # Only critical notifications during quiet hours
            return list(self._critical_queue)

        # Return by priority
        pending = []
        pending.extend(self._critical_queue)
        pending.extend(self._high_queue)
        pending.extend(self._normal_queue[:3])  # Limit normal

        return pending

    async def deliver(self, notification: Notification) -> bool:
        """Deliver a notification via TTS."""
        if not self.tts:
            logger.warning("No TTS callback configured")
            return False

        text = notification.spoken_text or notification.message

        try:
            await self.tts(text)
            notification.delivered = True
            notification.delivered_at = datetime.now()
            self._delivered.append(notification)

            # Remove from queues
            self._remove_from_queues(notification.id)

            return True

        except Exception as e:
            logger.error(f"Notification delivery failed: {e}")
            return False

    def acknowledge(self, notification_id: str) -> bool:
        """Mark notification as acknowledged."""
        for notif in self._delivered:
            if notif.id == notification_id:
                notif.acknowledged = True
                return True
        return False

    def _is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        now = datetime.now().time()

        if self.quiet_start <= self.quiet_end:
            # Simple range (e.g., 09:00 to 17:00)
            return self.quiet_start <= now <= self.quiet_end
        # Overnight range (e.g., 22:00 to 07:00)
        return now >= self.quiet_start or now <= self.quiet_end

    def _remove_from_queues(self, notification_id: str) -> None:
        """Remove notification from all queues."""
        for queue in [self._critical_queue, self._high_queue,
                      self._normal_queue, self._low_queue]:
            queue[:] = [n for n in queue if n.id != notification_id]


# =============================================================================
# Preference Learning
# =============================================================================

@dataclass
class UserPreference:
    """A learned user preference."""
    key: str
    value: Any
    confidence: float = 0.5
    observation_count: int = 1
    last_observed: datetime = field(default_factory=datetime.now)
    source: str = "inferred"


class PreferenceLearner:
    """
    Learns and applies user preferences from interactions.

    Tracks:
    - Communication preferences (email vs text)
    - Time preferences (morning meetings, evening reminders)
    - Contact preferences (frequent contacts)
    - Action patterns (common commands)
    """

    def __init__(
        self,
        ww_client: Any,
        min_observations: int = 3,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize preference learner.

        Args:
            ww_client: World Weaver MCP client
            min_observations: Minimum observations before applying
            confidence_threshold: Confidence needed to apply preference
        """
        self.ww = ww_client
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold

        # In-memory preferences (loaded from WW on start)
        self._preferences: dict[str, UserPreference] = {}

        # Observation buffer
        self._observations: list[dict] = []

    async def load_preferences(self) -> None:
        """Load preferences from World Weaver."""
        try:
            result = await self.ww.call_tool(
                "mcp__ww-memory__semantic_recall",
                {"query": "user preference", "limit": 50}
            )

            for ent in result.get("entities", []):
                pref = UserPreference(
                    key=ent.get("name"),
                    value=ent.get("value"),
                    confidence=ent.get("confidence", 0.5),
                )
                self._preferences[pref.key] = pref

            logger.info(f"Loaded {len(self._preferences)} preferences")

        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")

    def observe(
        self,
        action: str,
        parameters: dict,
        context: dict,
    ) -> None:
        """
        Observe a user action to learn from.

        Args:
            action: Action name (e.g., "calendar.create")
            parameters: Action parameters
            context: Context (time, location, etc.)
        """
        observation = {
            "action": action,
            "parameters": parameters,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self._observations.append(observation)

        # Trigger learning if enough observations
        if len(self._observations) >= 10:
            asyncio.create_task(self._learn_from_observations())

    async def _learn_from_observations(self) -> None:
        """Learn preferences from buffered observations."""
        observations = self._observations
        self._observations = []

        # Group by action type
        by_action: dict[str, list[dict]] = {}
        for obs in observations:
            action = obs["action"]
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(obs)

        # Learn patterns
        for action, obs_list in by_action.items():
            await self._learn_action_patterns(action, obs_list)

    async def _learn_action_patterns(
        self,
        action: str,
        observations: list[dict],
    ) -> None:
        """Learn patterns for a specific action type."""
        if len(observations) < self.min_observations:
            return

        # Time preference
        hours = [
            datetime.fromisoformat(obs["timestamp"]).hour
            for obs in observations
        ]
        avg_hour = sum(hours) / len(hours)

        # If consistent time preference
        hour_variance = sum((h - avg_hour) ** 2 for h in hours) / len(hours)
        if hour_variance < 4:  # Within ~2 hours
            pref_key = f"{action}_preferred_hour"
            await self._update_preference(pref_key, int(avg_hour), len(observations))

        # Parameter patterns (e.g., default recipients for email)
        if action.startswith("email"):
            recipients = [
                obs["parameters"].get("recipient", obs["parameters"].get("to"))
                for obs in observations
                if obs["parameters"].get("recipient") or obs["parameters"].get("to")
            ]
            if recipients:
                # Find most common
                from collections import Counter
                most_common = Counter(recipients).most_common(1)
                if most_common[0][1] >= self.min_observations:
                    await self._update_preference(
                        "frequent_email_contact",
                        most_common[0][0],
                        most_common[0][1],
                    )

    async def _update_preference(
        self,
        key: str,
        value: Any,
        observation_count: int,
    ) -> None:
        """Update or create a preference."""
        if key in self._preferences:
            pref = self._preferences[key]
            pref.observation_count += observation_count
            pref.value = value
            pref.confidence = min(1.0, pref.observation_count / 10)
            pref.last_observed = datetime.now()
        else:
            pref = UserPreference(
                key=key,
                value=value,
                observation_count=observation_count,
                confidence=observation_count / 10,
            )
            self._preferences[key] = pref

        # Store in WW if confident enough
        if pref.confidence >= self.confidence_threshold:
            await self._store_preference(pref)

    async def _store_preference(self, preference: UserPreference) -> None:
        """Store preference in World Weaver."""
        try:
            await self.ww.call_tool(
                "mcp__ww-memory__create_entity",
                {
                    "name": preference.key,
                    "type": "user_preference",
                    "summary": f"Value: {preference.value}, Confidence: {preference.confidence:.2f}",
                    "properties": {
                        "value": preference.value,
                        "confidence": preference.confidence,
                        "observation_count": preference.observation_count,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Failed to store preference: {e}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        pref = self._preferences.get(key)
        if pref and pref.confidence >= self.confidence_threshold:
            return pref.value
        return default

    def suggest_parameters(
        self,
        action: str,
        current_params: dict,
    ) -> dict:
        """Suggest parameters based on learned preferences."""
        suggestions = {}

        # Time-based suggestions
        time_pref = self.get_preference(f"{action}_preferred_hour")
        if time_pref and "when" not in current_params:
            suggestions["suggested_hour"] = time_pref

        # Contact suggestions for email
        if action.startswith("email") and "recipient" not in current_params:
            freq_contact = self.get_preference("frequent_email_contact")
            if freq_contact:
                suggestions["suggested_recipient"] = freq_contact

        return suggestions


# =============================================================================
# Multi-Modal Context
# =============================================================================

class ContentType(str, Enum):
    """Types of content that can be referenced."""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    URL = "url"
    CODE = "code"
    LOCATION = "location"


@dataclass
class ContextItem:
    """An item in the multi-modal context."""
    id: str
    content_type: ContentType
    content: Any  # Text, path, URL, etc.
    description: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


class MultiModalContext:
    """
    Handles non-voice content in voice interactions.

    Supports:
    - Images (from clipboard, screenshots)
    - Files (recently opened)
    - URLs (browsing context)
    - Code snippets (from editor)
    - Location (GPS if available)
    """

    def __init__(
        self,
        ww_client: Any,
        embedding_provider: Any | None = None,
        max_context_items: int = 10,
    ):
        """
        Initialize multi-modal context.

        Args:
            ww_client: World Weaver MCP client
            embedding_provider: For semantic matching
            max_context_items: Max items to track
        """
        self.ww = ww_client
        self.embedder = embedding_provider
        self.max_items = max_context_items

        # Context stack
        self._items: list[ContextItem] = []

        # References from voice (e.g., "that image", "this file")
        self._pronoun_refs: dict[str, str] = {}  # pronoun -> item_id

    def add_item(self, item: ContextItem) -> None:
        """Add an item to context."""
        self._items.insert(0, item)

        # Trim to max size
        if len(self._items) > self.max_items:
            self._items = self._items[:self.max_items]

        # Update pronoun references
        type_refs = {
            ContentType.IMAGE: ["that image", "this image", "the image", "it"],
            ContentType.FILE: ["that file", "this file", "the file", "it"],
            ContentType.URL: ["that page", "this page", "the link", "it"],
            ContentType.CODE: ["that code", "this code", "the snippet"],
        }

        refs = type_refs.get(item.content_type, [])
        for ref in refs:
            self._pronoun_refs[ref.lower()] = item.id

        logger.debug(f"Added {item.content_type.value} to context: {item.id}")

    def add_image(
        self,
        path_or_url: str,
        description: str | None = None,
    ) -> ContextItem:
        """Add an image to context."""
        item = ContextItem(
            id=str(uuid4()),
            content_type=ContentType.IMAGE,
            content=path_or_url,
            description=description,
        )
        self.add_item(item)
        return item

    def add_file(
        self,
        path: str,
        description: str | None = None,
    ) -> ContextItem:
        """Add a file to context."""
        item = ContextItem(
            id=str(uuid4()),
            content_type=ContentType.FILE,
            content=path,
            description=description,
        )
        self.add_item(item)
        return item

    def add_url(
        self,
        url: str,
        title: str | None = None,
    ) -> ContextItem:
        """Add a URL to context."""
        item = ContextItem(
            id=str(uuid4()),
            content_type=ContentType.URL,
            content=url,
            description=title,
        )
        self.add_item(item)
        return item

    def add_code(
        self,
        code: str,
        language: str | None = None,
        file_path: str | None = None,
    ) -> ContextItem:
        """Add code snippet to context."""
        item = ContextItem(
            id=str(uuid4()),
            content_type=ContentType.CODE,
            content=code,
            description=f"{language or 'code'} snippet",
            metadata={
                "language": language,
                "file_path": file_path,
            },
        )
        self.add_item(item)
        return item

    def resolve_reference(self, text: str) -> ContextItem | None:
        """
        Resolve a pronoun reference to a context item.

        Args:
            text: Text containing reference (e.g., "analyze that image")

        Returns:
            Referenced ContextItem or None
        """
        text_lower = text.lower()

        # Check explicit references
        for ref, item_id in self._pronoun_refs.items():
            if ref in text_lower:
                return self.get_item(item_id)

        # Check for type references
        type_keywords = {
            ContentType.IMAGE: ["image", "picture", "photo", "screenshot"],
            ContentType.FILE: ["file", "document"],
            ContentType.URL: ["page", "site", "link", "url"],
            ContentType.CODE: ["code", "snippet", "function"],
        }

        for content_type, keywords in type_keywords.items():
            if any(kw in text_lower for kw in keywords):
                # Return most recent of that type
                for item in self._items:
                    if item.content_type == content_type:
                        return item

        return None

    def get_item(self, item_id: str) -> ContextItem | None:
        """Get item by ID."""
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def get_recent(
        self,
        content_type: ContentType | None = None,
        limit: int = 5,
    ) -> list[ContextItem]:
        """Get recent context items."""
        items = self._items

        if content_type:
            items = [i for i in items if i.content_type == content_type]

        return items[:limit]

    def describe_context(self) -> str:
        """Generate description of current context for prompt."""
        if not self._items:
            return ""

        parts = ["## Current Context"]

        for item in self._items[:5]:
            desc = item.description or item.content_type.value
            if item.content_type == ContentType.IMAGE:
                parts.append(f"- Image: {desc}")
            elif item.content_type == ContentType.FILE:
                parts.append(f"- File: {item.content}")
            elif item.content_type == ContentType.URL:
                parts.append(f"- Viewing: {item.content}")
            elif item.content_type == ContentType.CODE:
                lang = item.metadata.get("language", "")
                parts.append(f"- Code ({lang}): {len(item.content)} chars")

        return "\n".join(parts)

    def clear(self) -> None:
        """Clear context."""
        self._items.clear()
        self._pronoun_refs.clear()


# =============================================================================
# Voice Triggers
# =============================================================================

@dataclass
class VoiceTrigger:
    """A custom voice trigger."""
    id: str
    phrases: list[str]  # Trigger phrases
    action: str  # Action to execute
    parameters: dict = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # Higher = checked first
    created_at: datetime = field(default_factory=datetime.now)


class VoiceTriggerManager:
    """
    Manages custom voice triggers.

    Allows users to define custom phrases that trigger specific actions.
    E.g., "good morning" -> read calendar + weather
    """

    def __init__(self):
        self._triggers: dict[str, VoiceTrigger] = {}
        self._phrase_index: dict[str, str] = {}  # phrase -> trigger_id

        # Register built-in triggers
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in triggers."""
        builtins = [
            VoiceTrigger(
                id="good_morning",
                phrases=["good morning", "morning briefing", "start my day"],
                action="routine.morning",
                priority=10,
            ),
            VoiceTrigger(
                id="good_night",
                phrases=["good night", "end of day", "shut down"],
                action="routine.evening",
                priority=10,
            ),
            VoiceTrigger(
                id="emergency",
                phrases=["emergency", "help me", "call for help"],
                action="emergency.alert",
                priority=100,
            ),
            VoiceTrigger(
                id="privacy_mode",
                phrases=["privacy mode", "stop listening", "go away"],
                action="system.privacy_mode",
                priority=90,
            ),
        ]

        for trigger in builtins:
            self.register(trigger)

    def register(self, trigger: VoiceTrigger) -> None:
        """Register a voice trigger."""
        self._triggers[trigger.id] = trigger

        for phrase in trigger.phrases:
            self._phrase_index[phrase.lower()] = trigger.id

        logger.debug(f"Registered trigger: {trigger.id}")

    def unregister(self, trigger_id: str) -> bool:
        """Unregister a trigger."""
        trigger = self._triggers.pop(trigger_id, None)
        if trigger:
            for phrase in trigger.phrases:
                self._phrase_index.pop(phrase.lower(), None)
            return True
        return False

    def match(self, text: str) -> VoiceTrigger | None:
        """
        Match text against registered triggers.

        Args:
            text: Voice input text

        Returns:
            Matched VoiceTrigger or None
        """
        text_lower = text.lower().strip()

        # Exact match
        if text_lower in self._phrase_index:
            trigger_id = self._phrase_index[text_lower]
            return self._triggers.get(trigger_id)

        # Partial match (text contains trigger phrase)
        matches = []
        for phrase, trigger_id in self._phrase_index.items():
            if phrase in text_lower:
                trigger = self._triggers.get(trigger_id)
                if trigger and trigger.enabled:
                    matches.append(trigger)

        if matches:
            # Return highest priority match
            return max(matches, key=lambda t: t.priority)

        return None

    def create_custom(
        self,
        phrases: list[str],
        action: str,
        parameters: dict | None = None,
    ) -> VoiceTrigger:
        """
        Create a custom trigger.

        Args:
            phrases: Trigger phrases
            action: Action to execute
            parameters: Action parameters

        Returns:
            Created VoiceTrigger
        """
        trigger = VoiceTrigger(
            id=str(uuid4()),
            phrases=phrases,
            action=action,
            parameters=parameters or {},
        )
        self.register(trigger)
        return trigger

    def get_all(self) -> list[VoiceTrigger]:
        """Get all registered triggers."""
        return list(self._triggers.values())
