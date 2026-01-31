"""Tests for Google Workspace integration."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.integrations.google_workspace import (
    GoogleWorkspaceSync,
    PersonalDataStore,
)
from t4dm.core.personal_entities import (
    Email,
    EmailAddress,
    EmailImportance,
    CalendarEvent,
    EventType,
    EventStatus,
    Contact,
    ContactType,
)


class TestGoogleWorkspaceSync:
    """Tests for GoogleWorkspaceSync class."""

    @pytest.fixture
    def sync(self):
        """Create sync instance with mocked MCP caller."""
        mcp_caller = AsyncMock()
        return GoogleWorkspaceSync(mcp_caller=mcp_caller)

    @pytest.mark.asyncio
    async def test_sync_emails_basic(self, sync):
        """Basic email sync returns emails."""
        sync.mcp.return_value = {
            "messages": [
                {
                    "id": "msg1",
                    "threadId": "thread1",
                    "from": "sender@example.com",
                    "to": "recipient@example.com",
                    "subject": "Test Subject",
                    "body": "Test body content",
                    "snippet": "Test snippet",
                    "date": "Mon, 1 Jan 2024 12:00:00 +0000",
                    "labelIds": ["INBOX"],
                }
            ]
        }

        emails = await sync.sync_emails(hours=24, max_results=10)
        assert len(emails) == 1
        assert emails[0].subject == "Test Subject"
        assert sync._last_email_sync is not None

    @pytest.mark.asyncio
    async def test_sync_emails_with_query(self, sync):
        """Email sync passes query parameter."""
        sync.mcp.return_value = {"messages": []}

        await sync.sync_emails(hours=24, query="is:unread")

        call_args = sync.mcp.call_args
        assert call_args[0][1]["query"] == "is:unread"

    @pytest.mark.asyncio
    async def test_sync_emails_empty(self, sync):
        """Empty email list is handled."""
        sync.mcp.return_value = {"messages": []}

        emails = await sync.sync_emails()
        assert emails == []

    @pytest.mark.asyncio
    async def test_process_email_with_name(self, sync):
        """Email address with name is parsed correctly."""
        sync.mcp.return_value = {
            "messages": [
                {
                    "id": "msg1",
                    "from": "John Doe <john@example.com>",
                    "to": "Jane <jane@example.com>",
                    "subject": "Test",
                    "body": "Content",
                    "date": "Mon, 1 Jan 2024 12:00:00 +0000",
                    "labelIds": [],
                }
            ]
        }

        emails = await sync.sync_emails()
        assert emails[0].from_address.name == "John Doe"
        assert emails[0].from_address.email == "john@example.com"

    @pytest.mark.asyncio
    async def test_process_email_important(self, sync):
        """Important label sets importance."""
        sync.mcp.return_value = {
            "messages": [
                {
                    "id": "msg1",
                    "from": "sender@example.com",
                    "to": "recipient@example.com",
                    "subject": "Important email",
                    "body": "Content",
                    "date": "Mon, 1 Jan 2024 12:00:00 +0000",
                    "labelIds": ["IMPORTANT"],
                }
            ]
        }

        emails = await sync.sync_emails()
        assert emails[0].importance == EmailImportance.HIGH

    @pytest.mark.asyncio
    async def test_process_email_action_required(self, sync):
        """Action words in subject set action_required."""
        sync.mcp.return_value = {
            "messages": [
                {
                    "id": "msg1",
                    "from": "sender@example.com",
                    "to": "recipient@example.com",
                    "subject": "Please respond urgently",
                    "body": "Content",
                    "date": "Mon, 1 Jan 2024 12:00:00 +0000",
                    "labelIds": [],
                }
            ]
        }

        emails = await sync.sync_emails()
        assert emails[0].action_required is True

    @pytest.mark.asyncio
    async def test_sync_calendar_basic(self, sync):
        """Basic calendar sync returns events."""
        sync.mcp.return_value = {
            "items": [
                {
                    "id": "event1",
                    "summary": "Team Meeting",
                    "start": {"dateTime": "2024-01-15T10:00:00Z"},
                    "end": {"dateTime": "2024-01-15T11:00:00Z"},
                    "status": "confirmed",
                }
            ]
        }

        events = await sync.sync_calendar(days=7)
        assert len(events) == 1
        assert events[0].title == "Team Meeting"
        assert sync._last_calendar_sync is not None

    @pytest.mark.asyncio
    async def test_sync_calendar_all_day(self, sync):
        """All-day events are parsed correctly."""
        sync.mcp.return_value = {
            "items": [
                {
                    "id": "event1",
                    "summary": "Holiday",
                    "start": {"date": "2024-01-01"},
                    "end": {"date": "2024-01-02"},
                    "status": "confirmed",
                }
            ]
        }

        events = await sync.sync_calendar()
        assert events[0].all_day is True

    @pytest.mark.asyncio
    async def test_classify_event_type_meeting(self, sync):
        """Meeting events are classified correctly."""
        event_type = sync._classify_event_type("Weekly standup meeting", {})
        assert event_type == EventType.MEETING

    @pytest.mark.asyncio
    async def test_classify_event_type_birthday(self, sync):
        """Birthday events are classified correctly."""
        event_type = sync._classify_event_type("John's Birthday", {})
        assert event_type == EventType.BIRTHDAY

    @pytest.mark.asyncio
    async def test_classify_event_type_deadline(self, sync):
        """Deadline events are classified correctly."""
        event_type = sync._classify_event_type("Project deadline", {})
        assert event_type == EventType.DEADLINE

    @pytest.mark.asyncio
    async def test_classify_event_type_focus(self, sync):
        """Focus time events are classified correctly."""
        # Use a phrase that doesn't contain other trigger words
        event_type = sync._classify_event_type("Deep work session", {})
        assert event_type == EventType.FOCUS_TIME

    @pytest.mark.asyncio
    async def test_classify_event_type_with_hangout(self, sync):
        """Events with hangout link are classified as meeting."""
        event_type = sync._classify_event_type(
            "Catch up",
            {"hangoutLink": "https://meet.google.com/abc-def-ghi"}
        )
        assert event_type == EventType.MEETING

    @pytest.mark.asyncio
    async def test_extract_contacts_from_emails(self, sync):
        """Contacts are extracted from email addresses."""
        emails = [
            Email(
                external_id="msg1",
                source="gmail",
                from_address=EmailAddress(name="John", email="john@company.com"),
                to_addresses=[EmailAddress(email="me@example.com")],
                subject="Test",
                date=datetime.now(),
            ),
            Email(
                external_id="msg2",
                source="gmail",
                from_address=EmailAddress(name="John", email="john@company.com"),
                to_addresses=[EmailAddress(email="me@example.com")],
                subject="Test 2",
                date=datetime.now(),
            ),
        ]

        contacts = await sync.extract_contacts_from_emails(emails)

        # John should appear once with interaction_count=2
        john = next(c for c in contacts if "john" in c.name.lower())
        assert john.interaction_count == 2

    @pytest.mark.asyncio
    async def test_extract_contacts_infers_organization(self, sync):
        """Organization is inferred from email domain."""
        emails = [
            Email(
                external_id="msg1",
                source="gmail",
                from_address=EmailAddress(email="jane@acmecorp.com"),
                to_addresses=[],
                subject="Test",
                date=datetime.now(),
            ),
        ]

        contacts = await sync.extract_contacts_from_emails(emails)
        jane = contacts[0]
        assert jane.organization == "Acmecorp"

    @pytest.mark.asyncio
    async def test_embed_emails_without_embedder(self, sync):
        """Embedding returns unmodified emails without embedder."""
        emails = [
            Email(
                external_id="msg1",
                source="gmail",
                from_address=EmailAddress(email="test@example.com"),
                to_addresses=[],
                subject="Test",
                date=datetime.now(),
            ),
        ]

        result = await sync.embed_emails(emails)
        assert result == emails

    @pytest.mark.asyncio
    async def test_embed_emails_with_embedder(self, sync):
        """Embedding adds vectors with embedder."""
        sync.embedder = MagicMock()
        sync.embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        emails = [
            Email(
                external_id="msg1",
                source="gmail",
                from_address=EmailAddress(email="test@example.com"),
                to_addresses=[],
                subject="Test",
                date=datetime.now(),
            ),
        ]

        result = await sync.embed_emails(emails)
        assert result[0].embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_build_personal_context(self, sync):
        """Personal context aggregates data."""
        # Mock calendar
        sync.mcp.side_effect = [
            {"items": []},  # Calendar
            {"messages": []},  # Emails
        ]

        ctx = await sync.build_personal_context()
        assert ctx.current_time is not None
        assert ctx.upcoming_events == []

    def test_parse_email_address_simple(self, sync):
        """Simple email address is parsed."""
        addr = sync._parse_email_address("test@example.com")
        assert addr.email == "test@example.com"
        assert addr.name is None

    def test_parse_email_address_with_name(self, sync):
        """Email with name is parsed."""
        addr = sync._parse_email_address("John Doe <john@example.com>")
        assert addr.email == "john@example.com"
        assert addr.name == "John Doe"

    def test_parse_date_valid(self, sync):
        """Valid date string is parsed."""
        date = sync._parse_date("Mon, 1 Jan 2024 12:00:00 +0000")
        assert date.year == 2024
        assert date.month == 1
        assert date.day == 1

    def test_parse_date_invalid(self, sync):
        """Invalid date returns current time."""
        date = sync._parse_date("not a date")
        assert date.year == datetime.now().year


class TestPersonalDataStore:
    """Tests for PersonalDataStore class."""

    @pytest.fixture
    def store(self):
        """Create store with mocked backends."""
        neo4j = AsyncMock()
        qdrant = AsyncMock()
        return PersonalDataStore(neo4j_store=neo4j, qdrant_store=qdrant)

    @pytest.mark.asyncio
    async def test_store_contact(self, store):
        """Contact is stored in Neo4j."""
        contact = Contact(
            name="John Doe",
            contact_type=ContactType.PERSON,
            source="test",
        )
        store.neo4j.create_node = AsyncMock(return_value="node-123")

        result = await store.store_contact(contact)
        assert result == "node-123"
        store.neo4j.create_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_contact_with_embedding(self, store):
        """Contact with embedding is stored in Qdrant."""
        contact = Contact(
            name="John Doe",
            contact_type=ContactType.PERSON,
            source="test",
            embedding=[0.1, 0.2, 0.3],
        )
        store.neo4j.create_node = AsyncMock(return_value="node-123")

        await store.store_contact(contact)
        store.qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_event(self, store):
        """Calendar event is stored."""
        event = CalendarEvent(
            external_id="event1",
            calendar_id="primary",
            source="google",
            title="Meeting",
            start=datetime.now(),
            end=datetime.now() + timedelta(hours=1),
        )
        store.neo4j.create_node = AsyncMock(return_value="node-456")

        result = await store.store_event(event)
        assert result == "node-456"

    @pytest.mark.asyncio
    async def test_store_email(self, store):
        """Email is stored."""
        email = Email(
            external_id="msg1",
            source="gmail",
            from_address=EmailAddress(email="test@example.com"),
            to_addresses=[],
            subject="Test",
            date=datetime.now(),
        )
        store.neo4j.create_node = AsyncMock(return_value="node-789")

        result = await store.store_email(email)
        assert result == "node-789"

    @pytest.mark.asyncio
    async def test_search_contacts_semantic(self, store):
        """Contact search uses Qdrant for semantic search."""
        store.qdrant.search = AsyncMock(return_value=[{"name": "John"}])
        store.neo4j.query = AsyncMock(return_value=[])

        results = await store.search_contacts(
            query="john",
            embedding=[0.1, 0.2, 0.3],
        )

        store.qdrant.search.assert_called_once()
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_contacts_by_name(self, store):
        """Contact search uses Neo4j for name search."""
        store.neo4j.query = AsyncMock(return_value=[{"c": {"name": "John"}}])

        results = await store.search_contacts(query="john")

        store.neo4j.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_events_with_dates(self, store):
        """Event search filters by date range."""
        store.neo4j.query = AsyncMock(return_value=[])

        await store.search_events(
            query="meeting",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )

        call_args = store.neo4j.query.call_args
        assert "start" in call_args[0][1]
        assert "end" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_upcoming_birthdays(self, store):
        """Birthday query returns contacts with dates."""
        store.neo4j.query = AsyncMock(return_value=[
            {"c": {"name": "John"}, "upcoming_birthday": datetime(2024, 1, 15)}
        ])

        results = await store.get_upcoming_birthdays(days=7)
        assert len(results) == 1
        assert results[0][0]["name"] == "John"


class TestPrivacyFiltering:
    """Tests for privacy filter integration."""

    @pytest.mark.asyncio
    async def test_blocked_email_not_returned(self):
        """Email blocked by privacy filter is not returned."""
        from t4dm.core.privacy_filter import PrivacyFilter, RedactionResult, SensitivityLevel

        mock_filter = MagicMock(spec=PrivacyFilter)
        mock_filter.filter.return_value = RedactionResult(
            original_length=100,
            redacted_length=0,
            content="",
            redactions=["SSN"],
            sensitivity=SensitivityLevel.RESTRICTED,
            blocked=True,
        )

        mcp = AsyncMock()
        mcp.return_value = {
            "messages": [
                {
                    "id": "msg1",
                    "from": "test@example.com",
                    "to": "me@example.com",
                    "subject": "Contains SSN 123-45-6789",
                    "body": "My SSN is 123-45-6789",
                    "date": "Mon, 1 Jan 2024 12:00:00 +0000",
                    "labelIds": [],
                }
            ]
        }

        sync = GoogleWorkspaceSync(mcp_caller=mcp, privacy_filter=mock_filter)
        emails = await sync.sync_emails()

        # Email should be filtered out
        assert len(emails) == 0


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_process_email_handles_error(self):
        """Email processing errors are handled gracefully."""
        mcp = AsyncMock()
        mcp.return_value = {
            "messages": [
                {"id": "msg1"},  # Missing required fields
            ]
        }

        sync = GoogleWorkspaceSync(mcp_caller=mcp)
        # Should set up to fetch full email but error handling catches it
        mcp.side_effect = [
            {"messages": [{"id": "msg1"}]},
            Exception("Failed to fetch email"),
        ]

        emails = await sync.sync_emails()
        # Error should be handled, empty result
        assert len(emails) == 0

    @pytest.mark.asyncio
    async def test_process_calendar_event_handles_error(self):
        """Calendar event processing errors are handled."""
        mcp = AsyncMock()
        mcp.return_value = {
            "items": [
                {"id": "event1"},  # Missing start/end
            ]
        }

        sync = GoogleWorkspaceSync(mcp_caller=mcp)
        events = await sync.sync_calendar()

        # Error should be handled
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_build_context_handles_calendar_error(self):
        """Personal context handles calendar errors."""
        mcp = AsyncMock()
        mcp.side_effect = Exception("Calendar API error")

        sync = GoogleWorkspaceSync(mcp_caller=mcp)
        ctx = await sync.build_personal_context()

        # Should still return context
        assert ctx.current_time is not None
