"""
Personal Data Integration for Kymera Voice.

Connects personal data (calendar, email, contacts, tasks) to voice interactions:
- Semantic search across personal data
- Birthday/anniversary reminders
- Email importance filtering
- Task management
- Contact lookup for voice commands
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from t4dm.core.personal_entities import (
    CalendarEvent,
    Contact,
    Email,
)
from t4dm.integrations.google_workspace import GoogleWorkspaceSync
from t4dm.integrations.kymera.bridge import VoiceContext

logger = logging.getLogger(__name__)


@dataclass
class PersonalDataConfig:
    """Configuration for personal data integration."""
    # Sync settings
    email_sync_hours: int = 24
    calendar_sync_days: int = 7
    auto_sync_interval_minutes: int = 15

    # Filtering
    email_max_results: int = 50
    ignore_promotional_emails: bool = True

    # Proactive notifications
    enable_birthday_reminders: bool = True
    birthday_reminder_days: int = 3
    enable_meeting_reminders: bool = True
    meeting_reminder_minutes: int = 15

    # Contact resolution
    fuzzy_name_matching: bool = True
    name_similarity_threshold: float = 0.7


class PersonalDataManager:
    """
    Manages personal data for voice assistant interactions.

    Provides:
    - Unified search across calendar, email, contacts
    - Contact resolution for voice commands
    - Proactive context (upcoming events, overdue tasks)
    - Birthday/anniversary tracking
    """

    def __init__(
        self,
        mcp_client: Any,
        embedding_provider: Any | None = None,
        config: PersonalDataConfig | None = None,
    ):
        """
        Initialize personal data manager.

        Args:
            mcp_client: MCP client for Google Workspace calls
            embedding_provider: Optional embedding provider for semantic search
            config: Optional configuration
        """
        self.mcp = mcp_client
        self.embedder = embedding_provider
        self.config = config or PersonalDataConfig()

        # Google Workspace sync
        self.workspace_sync = GoogleWorkspaceSync(
            mcp_caller=self._mcp_call,
            embedding_provider=embedding_provider,
        )

        # Cache
        self._contacts_cache: dict[str, Contact] = {}
        self._email_name_map: dict[str, str] = {}  # email -> name
        self._last_sync: datetime | None = None

        logger.info("PersonalDataManager initialized")

    async def _mcp_call(self, tool: str, params: dict) -> dict:
        """Call MCP tool."""
        return await self.mcp.call_tool(tool, params)

    # =========================================================================
    # Contact Resolution
    # =========================================================================

    async def resolve_contact(
        self,
        name_or_email: str,
        context: VoiceContext | None = None,
    ) -> Contact | None:
        """
        Resolve a name or email to a Contact.

        Handles:
        - Exact email match
        - Fuzzy name matching
        - Nickname resolution

        Args:
            name_or_email: Name or email to resolve
            context: Optional voice context for history

        Returns:
            Resolved Contact or None
        """
        query = name_or_email.lower().strip()

        # Check email format
        if "@" in query:
            return await self._resolve_by_email(query)

        # Try name resolution
        return await self._resolve_by_name(query)

    async def _resolve_by_email(self, email: str) -> Contact | None:
        """Resolve contact by email."""
        # Check cache
        if email in self._contacts_cache:
            return self._contacts_cache[email]

        # Search in memory
        try:
            result = await self.mcp.call_tool(
                "mcp__ww-memory__semantic_recall",
                {"query": f"contact email {email}", "limit": 1}
            )
            entities = result.get("entities", [])
            if entities:
                # Convert to Contact
                ent = entities[0]
                contact = Contact(
                    name=ent.get("name", email.split("@")[0]),
                    emails=[{"type": "email", "value": email, "primary": True}],
                )
                self._contacts_cache[email] = contact
                return contact

        except Exception as e:
            logger.debug(f"Contact lookup failed: {e}")

        return None

    async def _resolve_by_name(self, name: str) -> Contact | None:
        """Resolve contact by name with fuzzy matching."""
        # Try semantic search
        try:
            result = await self.mcp.call_tool(
                "mcp__ww-memory__semantic_recall",
                {"query": f"contact named {name}", "limit": 5}
            )

            for ent in result.get("entities", []):
                ent_name = ent.get("name", "").lower()

                # Exact match
                if ent_name == name:
                    return self._entity_to_contact(ent)

                # Fuzzy match
                if self.config.fuzzy_name_matching:
                    similarity = self._name_similarity(name, ent_name)
                    if similarity >= self.config.name_similarity_threshold:
                        return self._entity_to_contact(ent)

        except Exception as e:
            logger.debug(f"Name resolution failed: {e}")

        return None

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Simple name similarity score."""
        # Normalize
        n1 = name1.lower().split()
        n2 = name2.lower().split()

        # Check for matching parts
        matches = sum(1 for p1 in n1 if any(p1 in p2 or p2 in p1 for p2 in n2))

        return matches / max(len(n1), len(n2))

    def _entity_to_contact(self, entity: dict) -> Contact:
        """Convert WW entity to Contact."""
        return Contact(
            name=entity.get("name", "Unknown"),
            notes=entity.get("summary"),
            entity_id=entity.get("id"),
        )

    async def get_contact_email(self, name: str) -> str | None:
        """
        Get email address for a contact name.

        Convenience method for voice commands like "email John".
        """
        contact = await self.resolve_contact(name)
        if contact:
            return contact.primary_email()

        # Fallback: check email name map
        for email, mapped_name in self._email_name_map.items():
            if name.lower() in mapped_name.lower():
                return email

        return None

    # =========================================================================
    # Calendar Operations
    # =========================================================================

    async def get_todays_events(self) -> list[CalendarEvent]:
        """Get today's calendar events."""
        events = await self.workspace_sync.sync_calendar(days=1, include_past_days=0)
        today = date.today()

        return [
            e for e in events
            if e.start.date() == today
        ]

    async def get_upcoming_events(
        self,
        hours: int = 24,
    ) -> list[CalendarEvent]:
        """Get events in the next N hours."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        events = await self.workspace_sync.sync_calendar(days=2, include_past_days=0)

        return [
            e for e in events
            if now <= e.start <= cutoff
        ]

    async def get_next_event(self) -> CalendarEvent | None:
        """Get the next upcoming event."""
        events = await self.get_upcoming_events(hours=24)
        return events[0] if events else None

    async def search_events(
        self,
        query: str,
        days: int = 30,
    ) -> list[CalendarEvent]:
        """Search calendar events by title/description."""
        events = await self.workspace_sync.sync_calendar(days=days)
        query_lower = query.lower()

        return [
            e for e in events
            if query_lower in e.title.lower()
            or (e.description and query_lower in e.description.lower())
        ]

    async def format_events_for_voice(
        self,
        events: list[CalendarEvent],
        max_events: int = 5,
    ) -> str:
        """Format events for spoken response."""
        if not events:
            return "You have no events."

        parts = []
        for event in events[:max_events]:
            time_str = event.start.strftime("%I:%M %p")
            parts.append(f"At {time_str}, {event.title}")

        return ". ".join(parts)

    # =========================================================================
    # Email Operations
    # =========================================================================

    async def get_unread_emails(
        self,
        important_only: bool = False,
        max_results: int = 10,
    ) -> list[Email]:
        """Get unread emails."""
        query = "is:unread"
        if important_only:
            query += " is:important"

        emails = await self.workspace_sync.sync_emails(
            hours=self.config.email_sync_hours,
            max_results=max_results,
            query=query,
        )

        # Filter promotional if configured
        if self.config.ignore_promotional_emails:
            emails = [e for e in emails if "CATEGORY_PROMOTIONS" not in e.labels]

        return emails

    async def search_emails(
        self,
        query: str,
        from_person: str | None = None,
        max_results: int = 10,
    ) -> list[Email]:
        """Search emails."""
        search_query = query

        if from_person:
            # Resolve contact to email
            email = await self.get_contact_email(from_person)
            if email:
                search_query = f"from:{email} {query}"
            else:
                search_query = f"from:{from_person} {query}"

        return await self.workspace_sync.sync_emails(
            hours=168,  # 1 week
            max_results=max_results,
            query=search_query,
        )

    async def format_emails_for_voice(
        self,
        emails: list[Email],
        max_emails: int = 3,
    ) -> str:
        """Format emails for spoken response."""
        if not emails:
            return "You have no emails matching that."

        parts = []
        for email in emails[:max_emails]:
            sender = email.from_address.name or email.from_address.email.split("@")[0]
            parts.append(f"From {sender}: {email.subject}")

        count = len(emails)
        prefix = f"You have {count} emails. " if count > max_emails else ""

        return prefix + ". ".join(parts)

    # =========================================================================
    # Proactive Context
    # =========================================================================

    async def get_proactive_alerts(self) -> list[str]:
        """
        Get proactive alerts for the user.

        Checks:
        - Upcoming meetings (within meeting_reminder_minutes)
        - Overdue tasks
        - Upcoming birthdays
        - Important unread emails
        """
        alerts = []
        now = datetime.now()

        # Upcoming meetings
        if self.config.enable_meeting_reminders:
            events = await self.get_upcoming_events(hours=1)
            for event in events:
                mins_until = int((event.start - now).total_seconds() / 60)
                if 0 < mins_until <= self.config.meeting_reminder_minutes:
                    alerts.append(f"{event.title} starts in {mins_until} minutes")

        # Upcoming birthdays
        if self.config.enable_birthday_reminders:
            birthdays = await self.get_upcoming_birthdays()
            for name, bday in birthdays:
                days_until = (bday - date.today()).days
                if days_until == 0:
                    alerts.append(f"Today is {name}'s birthday!")
                elif days_until == 1:
                    alerts.append(f"Tomorrow is {name}'s birthday")
                elif days_until <= self.config.birthday_reminder_days:
                    alerts.append(f"{name}'s birthday is in {days_until} days")

        return alerts

    async def get_upcoming_birthdays(self) -> list[tuple[str, date]]:
        """Get contacts with upcoming birthdays."""
        try:
            # Query memory for contacts with birthdays
            result = await self.mcp.call_tool(
                "mcp__ww-memory__semantic_recall",
                {"query": "contact birthday", "limit": 50}
            )

            birthdays = []
            today = date.today()

            for ent in result.get("entities", []):
                bday_str = ent.get("birthday")
                if bday_str:
                    try:
                        bday = datetime.strptime(bday_str, "%Y-%m-%d").date()
                        # Adjust year to current/next year
                        this_year = bday.replace(year=today.year)
                        if this_year < today:
                            this_year = bday.replace(year=today.year + 1)

                        days_until = (this_year - today).days
                        if 0 <= days_until <= self.config.birthday_reminder_days:
                            birthdays.append((ent.get("name"), this_year))

                    except ValueError:
                        pass

            return sorted(birthdays, key=lambda x: x[1])

        except Exception as e:
            logger.debug(f"Birthday lookup failed: {e}")
            return []

    # =========================================================================
    # Voice Response Formatting
    # =========================================================================

    async def build_personal_context_summary(self) -> str:
        """Build summary of personal context for voice."""
        parts = []

        # Today's events
        events = await self.get_todays_events()
        if events:
            parts.append(f"Today: {len(events)} events")

        # Unread emails
        emails = await self.get_unread_emails(important_only=True, max_results=5)
        if emails:
            parts.append(f"Important unread: {len(emails)} emails")

        # Proactive alerts
        alerts = await self.get_proactive_alerts()
        if alerts:
            parts.append(f"Alerts: {len(alerts)}")

        return ". ".join(parts) if parts else "All clear"

    def format_time_for_voice(self, dt: datetime) -> str:
        """Format datetime for voice output."""
        datetime.now()
        today = date.today()

        if dt.date() == today:
            return f"at {dt.strftime('%I:%M %p')}"
        if dt.date() == today + timedelta(days=1):
            return f"tomorrow at {dt.strftime('%I:%M %p')}"
        if (dt.date() - today).days < 7:
            return f"on {dt.strftime('%A at %I:%M %p')}"
        return f"on {dt.strftime('%B %d at %I:%M %p')}"


class ContactCache:
    """
    Caches contact information for fast lookup.

    Built from email interactions and explicit contacts.
    """

    def __init__(self):
        self._by_email: dict[str, Contact] = {}
        self._by_name: dict[str, list[Contact]] = {}
        self._nicknames: dict[str, str] = {}  # nickname -> canonical name

    def add(self, contact: Contact) -> None:
        """Add contact to cache."""
        # Index by email
        for email_info in contact.emails:
            self._by_email[email_info.value.lower()] = contact

        # Index by name
        name_key = contact.name.lower()
        if name_key not in self._by_name:
            self._by_name[name_key] = []
        self._by_name[name_key].append(contact)

        # First name only
        first_name = contact.name.split()[0].lower() if contact.name else ""
        if first_name and first_name not in self._by_name:
            self._by_name[first_name] = []
        self._by_name[first_name].append(contact)

    def add_nickname(self, nickname: str, canonical_name: str) -> None:
        """Add nickname mapping."""
        self._nicknames[nickname.lower()] = canonical_name

    def find_by_email(self, email: str) -> Contact | None:
        """Find contact by email."""
        return self._by_email.get(email.lower())

    def find_by_name(self, name: str) -> list[Contact]:
        """Find contacts by name."""
        key = name.lower()

        # Check nickname
        if key in self._nicknames:
            key = self._nicknames[key].lower()

        return self._by_name.get(key, [])

    def clear(self) -> None:
        """Clear cache."""
        self._by_email.clear()
        self._by_name.clear()
        self._nicknames.clear()
