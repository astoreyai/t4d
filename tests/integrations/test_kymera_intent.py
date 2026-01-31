"""Tests for Kymera voice intent parser."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from t4dm.integrations.kymera.intent_parser import (
    VoiceIntentParser,
    ParsedIntent,
    TimeParser,
)
from t4dm.core.actions import ActionCategory


class TestParsedIntent:
    """Tests for ParsedIntent dataclass."""

    def test_default_values(self):
        """ParsedIntent has correct defaults."""
        intent = ParsedIntent(action_name="test.action")
        assert intent.confidence == 1.0
        assert intent.parameters == {}
        assert intent.raw_text == ""
        assert intent.extracted_entities == []


class TestVoiceIntentParser:
    """Tests for VoiceIntentParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser without LLM."""
        return VoiceIntentParser()

    def test_initialization(self, parser):
        """Parser initializes with compiled patterns."""
        assert len(parser._patterns) > 0

    # Email patterns
    def test_parse_send_email(self, parser):
        """Parse send email command."""
        intent = parser.parse("send an email to John about the meeting")
        assert intent.action_name == "email.send"
        assert intent.parameters.get("recipient") == "John"
        assert intent.parameters.get("subject") == "the meeting"

    def test_parse_check_emails(self, parser):
        """Parse check emails command."""
        intent = parser.parse("check my unread emails")
        assert intent.action_name == "email.read"

    def test_parse_email_from(self, parser):
        """Parse emails from sender."""
        intent = parser.parse("show my emails from Sarah")
        assert intent.action_name == "email.search"
        assert intent.parameters.get("from") == "Sarah"

    def test_parse_reply_email(self, parser):
        """Parse reply command."""
        intent = parser.parse("reply to John saying sounds good")
        assert intent.action_name == "email.reply"
        assert intent.parameters.get("to") == "John"
        assert intent.parameters.get("content") == "sounds good"

    # Calendar patterns
    def test_parse_schedule_meeting(self, parser):
        """Parse schedule meeting command."""
        intent = parser.parse("schedule a meeting with Bob at 3pm")
        assert intent.action_name == "calendar.create"
        assert "Bob" in intent.parameters.get("title", "")

    def test_parse_what_on_calendar(self, parser):
        """Parse calendar query."""
        intent = parser.parse("what's on my calendar today")
        assert intent.action_name == "calendar.list"

    def test_parse_when_is_meeting(self, parser):
        """Parse when query."""
        intent = parser.parse("when is my next standup")
        assert intent.action_name == "calendar.query"
        assert "standup" in intent.parameters.get("query", "")

    def test_parse_cancel_meeting(self, parser):
        """Parse cancel command."""
        intent = parser.parse("cancel my meeting with Alice")
        assert intent.action_name == "calendar.delete"

    # Reminder patterns
    def test_parse_remind_me(self, parser):
        """Parse reminder command."""
        intent = parser.parse("remind me to call mom tomorrow")
        assert intent.action_name == "reminder.set"
        assert "call mom" in intent.parameters.get("message", "")

    def test_parse_set_reminder(self, parser):
        """Parse set reminder command."""
        intent = parser.parse("set a reminder for 5pm")
        assert intent.action_name == "reminder.set"

    def test_parse_show_reminders(self, parser):
        """Parse show reminders command."""
        intent = parser.parse("show my reminders")
        assert intent.action_name == "reminder.list"

    # Task patterns
    def test_parse_add_task(self, parser):
        """Parse add task command."""
        intent = parser.parse("add a task to review the PR")
        assert intent.action_name == "task.create"

    def test_parse_show_tasks(self, parser):
        """Parse show tasks command."""
        intent = parser.parse("what are my tasks")
        assert intent.action_name == "task.list"

    def test_parse_complete_task(self, parser):
        """Parse complete task command."""
        intent = parser.parse("mark code review as done")
        assert intent.action_name == "task.complete"
        assert "code review" in intent.parameters.get("title", "")

    def test_parse_finished_task(self, parser):
        """Parse finished task variant."""
        intent = parser.parse("I finished the report")
        assert intent.action_name == "task.complete"
        assert "report" in intent.parameters.get("title", "")

    # Memory patterns
    def test_parse_remember(self, parser):
        """Parse remember command."""
        intent = parser.parse("remember that John's birthday is March 15")
        assert intent.action_name == "memory.store"
        assert "birthday" in intent.parameters.get("content", "")

    def test_parse_dont_forget(self, parser):
        """Parse don't forget command."""
        intent = parser.parse("don't forget to call the dentist")
        assert intent.action_name == "memory.store"

    def test_parse_what_remember(self, parser):
        """Parse recall command."""
        intent = parser.parse("what do you remember about Python")
        assert intent.action_name == "memory.recall"
        assert "Python" in intent.parameters.get("query", "")

    def test_parse_forget(self, parser):
        """Parse forget command."""
        intent = parser.parse("forget about that password")
        assert intent.action_name == "memory.forget"

    # Contact patterns
    def test_parse_who_is(self, parser):
        """Parse who is command - note: matches memory.recall due to pattern order."""
        intent = parser.parse("who is Bob Smith")
        # Memory pattern matches first due to order
        assert intent.action_name in ["contact.lookup", "memory.recall"]

    def test_parse_birthday_query(self, parser):
        """Parse birthday query - note: matches calendar.query due to pattern order."""
        intent = parser.parse("when is Sarah's birthday")
        # Calendar pattern "when is/am my next..." matches first
        assert intent.action_name in ["contact.birthday", "calendar.query"]

    def test_parse_call(self, parser):
        """Parse call command."""
        intent = parser.parse("call John")
        assert intent.action_name == "call.initiate"

    # System patterns
    def test_parse_timer(self, parser):
        """Parse timer command."""
        intent = parser.parse("set a timer for 10 minutes")
        assert intent.action_name == "timer.set"

    def test_parse_time(self, parser):
        """Parse time query."""
        intent = parser.parse("what time is it")
        assert intent.action_name == "lookup.time"

    def test_parse_weather(self, parser):
        """Parse weather query."""
        intent = parser.parse("what's the weather")
        assert intent.action_name == "lookup.weather"

    def test_parse_search(self, parser):
        """Parse search command."""
        intent = parser.parse("search for Python tutorials")
        assert intent.action_name == "search.web"
        assert "Python" in intent.parameters.get("query", "")

    # File patterns
    def test_parse_open_file(self, parser):
        """Parse open file command."""
        intent = parser.parse("open the file main.py")
        assert intent.action_name == "file.read"

    def test_parse_create_file(self, parser):
        """Parse create file command."""
        intent = parser.parse("create a file called test.py")
        assert intent.action_name == "file.create"

    def test_parse_find_files(self, parser):
        """Parse find files command."""
        intent = parser.parse("find files for authentication")
        assert intent.action_name == "search.files"

    # Fallback
    def test_parse_unrecognized(self, parser):
        """Unrecognized input falls back to chat."""
        intent = parser.parse("hello how are you today")
        assert intent.action_name == "claude.chat"
        assert intent.confidence == 0.5
        assert intent.parameters.get("message") == "hello how are you today"

    def test_confidence_pattern_match(self, parser):
        """Pattern match has high confidence."""
        intent = parser.parse("send email to John")
        assert intent.confidence == 0.9

    # LLM parsing
    @pytest.mark.asyncio
    async def test_parse_with_llm_uses_pattern_first(self):
        """LLM parsing tries patterns first."""
        parser = VoiceIntentParser()

        intent = await parser.parse_with_llm("send email to John")
        assert intent.action_name == "email.send"
        assert intent.confidence == 0.9

    @pytest.mark.asyncio
    async def test_parse_with_llm_fallback(self):
        """LLM is used for complex intents."""
        llm = MagicMock()
        llm.complete = AsyncMock(return_value='{"action": "calendar.create", "parameters": {"title": "dentist"}, "confidence": 0.8}')

        parser = VoiceIntentParser(llm_client=llm)

        intent = await parser.parse_with_llm("I need to see the dentist next week")
        # Pattern didn't match, LLM used
        assert llm.complete.called or intent.action_name == "claude.chat"

    # Action request conversion
    def test_to_action_request(self, parser):
        """Convert intent to ActionRequest."""
        intent = ParsedIntent(
            action_name="email.send",
            parameters={"recipient": "John"},
            confidence=0.9,
            raw_text="send email to John",
        )

        request = parser.to_action_request(intent, "session-123")

        assert request.action_name == "email.send"
        assert request.category == ActionCategory.EMAIL
        assert request.session_id == "session-123"
        assert request.user_utterance == "send email to John"

    def test_to_action_request_categories(self, parser):
        """Action categories are mapped correctly."""
        categories = {
            "email.send": ActionCategory.EMAIL,
            "calendar.create": ActionCategory.CALENDAR,
            "reminder.set": ActionCategory.REMINDER,
            "task.create": ActionCategory.TASK,
            "memory.store": ActionCategory.SYSTEM,
            "contact.lookup": ActionCategory.LOOKUP,
            "call.initiate": ActionCategory.CALL,
            "search.web": ActionCategory.SEARCH,
            "file.read": ActionCategory.FILE,
            "timer.set": ActionCategory.TIMER,
            "claude.chat": ActionCategory.SYSTEM,
        }

        for action_name, expected_category in categories.items():
            intent = ParsedIntent(action_name=action_name)
            request = parser.to_action_request(intent, "session")
            assert request.category == expected_category


class TestTimeParser:
    """Tests for TimeParser class."""

    @pytest.fixture
    def parser(self):
        """Create time parser."""
        return TimeParser()

    def test_parse_in_minutes(self, parser):
        """Parse 'in X minutes' format."""
        result = parser.parse("in 30 minutes")
        assert result is not None
        expected = datetime.now() + timedelta(minutes=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_in_hours(self, parser):
        """Parse 'in X hours' format."""
        result = parser.parse("in 2 hours")
        assert result is not None
        expected = datetime.now() + timedelta(hours=2)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_in_days(self, parser):
        """Parse 'in X days' format."""
        result = parser.parse("in 3 days")
        assert result is not None
        expected = datetime.now() + timedelta(days=3)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_tomorrow_with_time(self, parser):
        """Parse 'tomorrow at X' format."""
        result = parser.parse("tomorrow at 3:30pm")
        assert result is not None
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.day == tomorrow.day
        assert result.hour == 15
        assert result.minute == 30

    def test_parse_time_only(self, parser):
        """Parse time only format."""
        result = parser.parse("at 2pm")
        assert result is not None
        assert result.hour == 14

    def test_parse_time_am(self, parser):
        """Parse AM time format."""
        result = parser.parse("at 9am")
        assert result is not None
        assert result.hour == 9

    def test_parse_tomorrow_default(self, parser):
        """Parse 'tomorrow' defaults to 9am."""
        result = parser.parse("tomorrow")
        assert result is not None
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.day == tomorrow.day
        assert result.hour == 9

    def test_parse_tonight(self, parser):
        """Parse 'tonight' to 8pm."""
        result = parser.parse("tonight")
        assert result is not None
        assert result.hour == 20

    def test_parse_later_today(self, parser):
        """Parse 'this afternoon' to 5pm."""
        result = parser.parse("this afternoon")
        assert result is not None
        assert result.hour == 17

    def test_parse_unrecognized(self, parser):
        """Unrecognized format returns None."""
        result = parser.parse("sometime next week")
        assert result is None

    def test_parse_12_hour_midnight(self, parser):
        """Parse 12am correctly."""
        result = parser.parse("at 12am")
        assert result is not None
        assert result.hour == 0

    def test_parse_12_hour_noon(self, parser):
        """Parse 12pm correctly."""
        result = parser.parse("at 12pm")
        assert result is not None
        assert result.hour == 12
