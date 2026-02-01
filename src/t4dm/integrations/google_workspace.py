"""
Google Workspace Integration for World Weaver.

Syncs calendars, contacts, and emails from Google into World Weaver
for semantic search and contextual retrieval.

Uses MCP tools:
- mcp__google-workspace__gmail_*
- mcp__google-workspace__calendar_*
- mcp__google-workspace__drive_*
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from t4dm.core.personal_entities import (
    CalendarEvent,
    Contact,
    ContactInfo,
    ContactType,
    Email,
    EmailAddress,
    EmailCategory,
    EmailImportance,
    EventAttendee,
    EventStatus,
    EventType,
    PersonalContext,
)
from t4dm.core.privacy_filter import PrivacyFilter

logger = logging.getLogger(__name__)


class GoogleWorkspaceSync:
    """
    Synchronizes Google Workspace data with World Weaver.

    Handles:
    - Gmail messages → Email entities
    - Calendar events → CalendarEvent entities
    - Contacts → Contact entities
    - Drive files → searchable embeddings
    """

    def __init__(
        self,
        mcp_caller: Any,  # Function to call MCP tools
        privacy_filter: PrivacyFilter | None = None,
        embedding_provider: Any = None,
    ):
        """
        Initialize sync.

        Args:
            mcp_caller: Async function to call MCP tools
            privacy_filter: Optional privacy filter for content
            embedding_provider: Embedding provider for vectors
        """
        self.mcp = mcp_caller
        self.privacy = privacy_filter or PrivacyFilter()
        self.embedder = embedding_provider

        # Sync state
        self._last_email_sync: datetime | None = None
        self._last_calendar_sync: datetime | None = None
        self._last_contact_sync: datetime | None = None

    # =========================================================================
    # Email Sync
    # =========================================================================

    async def sync_emails(
        self,
        hours: int = 24,
        max_results: int = 50,
        query: str | None = None,
    ) -> list[Email]:
        """
        Sync recent emails from Gmail.

        Args:
            hours: How many hours back to sync
            max_results: Maximum emails to fetch
            query: Optional Gmail search query

        Returns:
            List of Email entities
        """
        logger.info(f"Syncing emails from last {hours} hours")

        # Fetch from Gmail
        params = {
            "hours": hours,
            "maxResults": max_results,
        }
        if query:
            params["query"] = query

        result = await self.mcp("mcp__google-workspace__gmail_list_emails", params)

        emails = []
        for msg in result.get("messages", []):
            email = await self._process_email(msg)
            if email:
                emails.append(email)

        self._last_email_sync = datetime.now()
        logger.info(f"Synced {len(emails)} emails")
        return emails

    async def _process_email(self, raw: dict) -> Email | None:
        """Convert Gmail message to Email entity."""
        try:
            # Get full content if needed
            if "body" not in raw:
                full = await self.mcp(
                    "mcp__google-workspace__gmail_read_email",
                    {"emailId": raw["id"]}
                )
                raw.update(full)

            # Parse addresses
            from_addr = self._parse_email_address(raw.get("from", ""))
            to_addrs = [
                self._parse_email_address(a)
                for a in raw.get("to", "").split(",")
                if a.strip()
            ]

            # Apply privacy filter
            subject = raw.get("subject", "")
            body = raw.get("body", "") or raw.get("snippet", "")

            subject_result = self.privacy.filter(subject)
            body_result = self.privacy.filter(body)

            if subject_result.blocked and body_result.blocked:
                logger.debug(f"Email blocked by privacy filter: {raw['id']}")
                return None

            email = Email(
                external_id=raw["id"],
                thread_id=raw.get("threadId"),
                source="gmail",
                from_address=from_addr,
                to_addresses=to_addrs,
                subject=subject_result.content,
                snippet=raw.get("snippet", "")[:200],
                body_text=body_result.content if not body_result.blocked else None,
                date=self._parse_date(raw.get("date", "")),
                is_read="UNREAD" not in raw.get("labelIds", []),
                is_starred="STARRED" in raw.get("labelIds", []),
                labels=raw.get("labelIds", []),
                has_attachments=bool(raw.get("attachments")),
            )

            # Classify importance
            if "IMPORTANT" in email.labels:
                email.importance = EmailImportance.HIGH
            elif "CATEGORY_PROMOTIONS" in email.labels:
                email.importance = EmailImportance.LOW
                email.category = EmailCategory.PROMOTIONS

            # Check if action required (simple heuristic)
            action_words = ["please", "urgent", "asap", "need", "action", "respond"]
            if any(w in email.subject.lower() for w in action_words):
                email.action_required = True

            return email

        except Exception as e:
            logger.error(f"Failed to process email: {e}")
            return None

    def _parse_email_address(self, raw: str) -> EmailAddress:
        """Parse 'Name <email@example.com>' format."""
        import re
        match = re.match(r"(.+?)\s*<(.+?)>", raw.strip())
        if match:
            return EmailAddress(name=match.group(1).strip(), email=match.group(2))
        return EmailAddress(email=raw.strip())

    def _parse_date(self, raw: str) -> datetime:
        """Parse email date string."""
        from email.utils import parsedate_to_datetime
        try:
            return parsedate_to_datetime(raw)
        except Exception:
            return datetime.now()

    # =========================================================================
    # Calendar Sync
    # =========================================================================

    async def sync_calendar(
        self,
        days: int = 7,
        calendar_id: str = "primary",
        include_past_days: int = 1,
    ) -> list[CalendarEvent]:
        """
        Sync calendar events.

        Args:
            days: Days forward to sync
            calendar_id: Calendar ID
            include_past_days: Also include recent past events

        Returns:
            List of CalendarEvent entities
        """
        logger.info(f"Syncing calendar for next {days} days")

        # Calculate date range
        start_date = (datetime.now() - timedelta(days=include_past_days)).strftime("%Y-%m-%d")

        result = await self.mcp(
            "mcp__google-workspace__calendar_list_events",
            {
                "calendarId": calendar_id,
                "date": start_date,
                "days": days + include_past_days,
                "maxResults": 100,
            }
        )

        events = []
        for raw in result.get("items", []):
            event = self._process_calendar_event(raw)
            if event:
                events.append(event)

        self._last_calendar_sync = datetime.now()
        logger.info(f"Synced {len(events)} calendar events")
        return events

    def _process_calendar_event(self, raw: dict) -> CalendarEvent | None:
        """Convert Google Calendar event to CalendarEvent entity."""
        try:
            # Parse times
            start_raw = raw.get("start", {})
            end_raw = raw.get("end", {})

            if "dateTime" in start_raw:
                start = datetime.fromisoformat(start_raw["dateTime"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_raw["dateTime"].replace("Z", "+00:00"))
                all_day = False
            else:
                # All-day event
                start = datetime.strptime(start_raw.get("date", ""), "%Y-%m-%d")
                end = datetime.strptime(end_raw.get("date", ""), "%Y-%m-%d")
                all_day = True

            # Parse attendees
            attendees = []
            for att in raw.get("attendees", []):
                attendees.append(EventAttendee(
                    email=att.get("email", ""),
                    name=att.get("displayName"),
                    response=att.get("responseStatus", "needsAction"),
                    organizer=att.get("organizer", False),
                ))

            # Determine event type
            summary = raw.get("summary", "")
            event_type = self._classify_event_type(summary, raw)

            # Status
            status_map = {
                "confirmed": EventStatus.CONFIRMED,
                "tentative": EventStatus.TENTATIVE,
                "cancelled": EventStatus.CANCELLED,
            }
            status = status_map.get(raw.get("status", ""), EventStatus.CONFIRMED)

            event = CalendarEvent(
                external_id=raw["id"],
                calendar_id=raw.get("calendarId", "primary"),
                source="google",
                title=summary,
                description=raw.get("description"),
                event_type=event_type,
                status=status,
                start=start,
                end=end,
                all_day=all_day,
                timezone=start_raw.get("timeZone", "UTC"),
                location=raw.get("location"),
                meeting_link=raw.get("hangoutLink"),
                attendees=attendees,
                organizer_email=raw.get("organizer", {}).get("email"),
                recurring=bool(raw.get("recurringEventId")),
                recurrence_rule=raw.get("recurrence", [None])[0] if raw.get("recurrence") else None,
            )

            return event

        except Exception as e:
            logger.error(f"Failed to process calendar event: {e}")
            return None

    def _classify_event_type(self, summary: str, raw: dict) -> EventType:
        """Classify event type from summary and metadata."""
        summary_lower = summary.lower()

        if "birthday" in summary_lower:
            return EventType.BIRTHDAY
        if "anniversary" in summary_lower:
            return EventType.ANNIVERSARY
        if any(w in summary_lower for w in ["deadline", "due"]):
            return EventType.DEADLINE
        if any(w in summary_lower for w in ["reminder", "remind"]):
            return EventType.REMINDER
        if any(w in summary_lower for w in ["meeting", "call", "sync", "standup"]):
            return EventType.MEETING
        if any(w in summary_lower for w in ["focus", "deep work", "no meetings"]):
            return EventType.FOCUS_TIME
        if any(w in summary_lower for w in ["flight", "travel", "trip"]):
            return EventType.TRAVEL

        # Check for video conference
        if raw.get("hangoutLink") or "zoom" in summary_lower or "meet" in summary_lower:
            return EventType.MEETING

        return EventType.OTHER

    # =========================================================================
    # Contact Sync (via search, as Contacts API not in MCP)
    # =========================================================================

    async def extract_contacts_from_emails(
        self,
        emails: list[Email],
    ) -> list[Contact]:
        """
        Extract contacts from email addresses.

        Since direct Contacts API may not be available, we extract
        from email headers.
        """
        seen: dict[str, Contact] = {}

        for email in emails:
            # Process all addresses
            addresses = (
                [email.from_address] +
                email.to_addresses +
                email.cc_addresses
            )

            for addr in addresses:
                if addr.email in seen:
                    # Update interaction count
                    seen[addr.email].interaction_count += 1
                    seen[addr.email].last_contacted = max(
                        seen[addr.email].last_contacted or datetime.min,
                        email.date,
                    )
                else:
                    contact = Contact(
                        name=addr.name or addr.email.split("@")[0],
                        contact_type=ContactType.PERSON,
                        emails=[ContactInfo(type="email", value=addr.email, primary=True)],
                        source="email_extraction",
                        interaction_count=1,
                        last_contacted=email.date,
                    )

                    # Infer organization from email domain
                    domain = addr.email.split("@")[-1]
                    if domain not in ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]:
                        contact.organization = domain.split(".")[0].title()

                    seen[addr.email] = contact

        return list(seen.values())

    # =========================================================================
    # Embedding Generation
    # =========================================================================

    async def embed_emails(self, emails: list[Email]) -> list[Email]:
        """Generate embeddings for emails."""
        if not self.embedder:
            return emails

        texts = [e.summary_for_embedding() for e in emails]
        embeddings = await self.embedder.embed_batch(texts)

        for email, embedding in zip(emails, embeddings):
            email.embedding = embedding

        return emails

    async def embed_events(self, events: list[CalendarEvent]) -> list[CalendarEvent]:
        """Generate embeddings for calendar events."""
        if not self.embedder:
            return events

        texts = [e.summary_for_embedding() for e in events]
        embeddings = await self.embedder.embed_batch(texts)

        for event, embedding in zip(events, embeddings):
            event.embedding = embedding

        return events

    async def embed_contacts(self, contacts: list[Contact]) -> list[Contact]:
        """Generate embeddings for contacts."""
        if not self.embedder:
            return contacts

        texts = [c.summary_for_embedding() for c in contacts]
        embeddings = await self.embedder.embed_batch(texts)

        for contact, embedding in zip(contacts, embeddings):
            contact.embedding = embedding

        return contacts

    # =========================================================================
    # Personal Context Building
    # =========================================================================

    async def build_personal_context(
        self,
        current_time: datetime | None = None,
    ) -> PersonalContext:
        """
        Build aggregated personal context.

        Fetches and combines:
        - Upcoming calendar events
        - Overdue/due tasks
        - Important unread emails
        - Upcoming birthdays
        """
        now = current_time or datetime.now()
        ctx = PersonalContext(current_time=now)

        # Get calendar events
        try:
            events = await self.sync_calendar(days=1, include_past_days=0)
            ctx.upcoming_events = [
                e for e in events
                if e.start >= now and e.start <= now + timedelta(hours=24)
            ]
        except Exception as e:
            logger.error(f"Failed to get calendar: {e}")

        # Get emails needing attention
        try:
            emails = await self.sync_emails(hours=48, max_results=20, query="is:unread")
            ctx.unread_important_emails = [
                e for e in emails
                if e.importance == EmailImportance.HIGH and not e.is_read
            ]
            ctx.action_required_emails = [
                e for e in emails
                if e.action_required
            ]
        except Exception as e:
            logger.error(f"Failed to get emails: {e}")

        return ctx


class PersonalDataStore:
    """
    Stores personal entities in World Weaver.

    Handles dual-store (Neo4j + Qdrant) persistence for:
    - Contacts
    - Calendar Events
    - Emails
    - Tasks
    """

    def __init__(
        self,
        graph_store: Any,  # T4DXGraphStore
        vector_store: Any,  # T4DXVectorStore
    ):
        self.neo4j = graph_store
        self.qdrant = vector_store

    async def store_contact(self, contact: Contact) -> str:
        """Store contact in knowledge graph."""
        # Create Neo4j node
        node_id = await self.neo4j.create_node(
            labels=["Contact", contact.contact_type.value],
            properties={
                "id": str(contact.id),
                "external_id": contact.external_id,
                "name": contact.name,
                "display_name": contact.display_name,
                "organization": contact.organization,
                "job_title": contact.job_title,
                "relationship": contact.relationship.value,
                "source": contact.source,
                "created_at": contact.created_at.isoformat(),
            }
        )

        # Store embedding in Qdrant if available
        if contact.embedding:
            await self.qdrant.upsert(
                collection="contacts",
                id=str(contact.id),
                vector=contact.embedding,
                payload={
                    "name": contact.name,
                    "organization": contact.organization,
                    "type": contact.contact_type.value,
                }
            )

        return node_id

    async def store_event(self, event: CalendarEvent) -> str:
        """Store calendar event."""
        node_id = await self.neo4j.create_node(
            labels=["CalendarEvent", event.event_type.value],
            properties={
                "id": str(event.id),
                "external_id": event.external_id,
                "title": event.title,
                "start": event.start.isoformat(),
                "end": event.end.isoformat(),
                "location": event.location,
                "event_type": event.event_type.value,
                "status": event.status.value,
            }
        )

        if event.embedding:
            await self.qdrant.upsert(
                collection="events",
                id=str(event.id),
                vector=event.embedding,
                payload={
                    "title": event.title,
                    "start": event.start.isoformat(),
                    "type": event.event_type.value,
                }
            )

        return node_id

    async def store_email(self, email: Email) -> str:
        """Store email."""
        node_id = await self.neo4j.create_node(
            labels=["Email"],
            properties={
                "id": str(email.id),
                "external_id": email.external_id,
                "thread_id": email.thread_id,
                "subject": email.subject,
                "from_email": email.from_address.email,
                "from_name": email.from_address.name,
                "date": email.date.isoformat(),
                "importance": email.importance.value,
                "action_required": email.action_required,
            }
        )

        if email.embedding:
            await self.qdrant.upsert(
                collection="emails",
                id=str(email.id),
                vector=email.embedding,
                payload={
                    "subject": email.subject,
                    "from": email.from_address.email,
                    "date": email.date.isoformat(),
                }
            )

        return node_id

    async def search_contacts(
        self,
        query: str,
        embedding: list[float] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search contacts by name or semantic similarity."""
        results = []

        # Semantic search if embedding provided
        if embedding:
            qdrant_results = await self.qdrant.search(
                collection="contacts",
                vector=embedding,
                limit=limit,
            )
            results.extend(qdrant_results)

        # Also do name search in Neo4j
        neo4j_results = await self.neo4j.query(
            """
            MATCH (c:Contact)
            WHERE toLower(c.name) CONTAINS toLower($query)
               OR toLower(c.organization) CONTAINS toLower($query)
            RETURN c
            LIMIT $limit
            """,
            {"query": query, "limit": limit}
        )
        results.extend(neo4j_results)

        return results

    async def search_events(
        self,
        query: str,
        embedding: list[float] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search calendar events."""
        results = []

        # Semantic search
        if embedding:
            qdrant_results = await self.qdrant.search(
                collection="events",
                vector=embedding,
                limit=limit,
            )
            results.extend(qdrant_results)

        # Date-filtered search in Neo4j
        params = {"query": query, "limit": limit}
        date_filter = ""

        if start_date:
            params["start"] = start_date.isoformat()
            date_filter += " AND e.start >= datetime($start)"
        if end_date:
            params["end"] = end_date.isoformat()
            date_filter += " AND e.start <= datetime($end)"

        neo4j_results = await self.neo4j.query(
            f"""
            MATCH (e:CalendarEvent)
            WHERE toLower(e.title) CONTAINS toLower($query) {date_filter}
            RETURN e
            ORDER BY e.start
            LIMIT $limit
            """,
            params
        )
        results.extend(neo4j_results)

        return results

    async def get_upcoming_birthdays(
        self,
        days: int = 7,
    ) -> list[tuple[dict, datetime]]:
        """Get contacts with upcoming birthdays."""
        datetime.now()

        results = await self.neo4j.query(
            """
            MATCH (c:Contact)
            WHERE c.birthday IS NOT NULL
            WITH c,
                 date(c.birthday) AS bday,
                 date() AS today
            WITH c,
                 date({year: today.year, month: bday.month, day: bday.day}) AS thisYear
            WHERE thisYear >= today AND thisYear <= today + duration({days: $days})
            RETURN c, thisYear AS upcoming_birthday
            ORDER BY thisYear
            """,
            {"days": days}
        )

        return [(r["c"], r["upcoming_birthday"]) for r in results]
