"""Tests for personal data entities."""

import pytest
from datetime import datetime, date, timedelta
from uuid import uuid4

from ww.core.personal_entities import (
    # Enums
    ContactType,
    RelationshipType,
    EventType,
    EventStatus,
    EmailImportance,
    EmailCategory,
    TaskPriority,
    TaskStatus,
    # Contact models
    ContactInfo,
    Contact,
    # Event models
    EventAttendee,
    EventReminder,
    CalendarEvent,
    # Email models
    EmailAddress,
    EmailAttachment,
    Email,
    # Task model
    Task,
    # Location model
    Location,
    # Context
    PersonalContext,
)


class TestContactTypeEnum:
    """Tests for ContactType enum."""

    def test_values(self):
        """All contact types exist."""
        assert ContactType.PERSON.value == "person"
        assert ContactType.ORGANIZATION.value == "organization"
        assert ContactType.GROUP.value == "group"
        assert ContactType.SERVICE.value == "service"

    def test_count(self):
        """All types accounted for."""
        assert len(list(ContactType)) == 4


class TestRelationshipTypeEnum:
    """Tests for RelationshipType enum."""

    def test_values(self):
        """All relationship types exist."""
        assert RelationshipType.FAMILY.value == "family"
        assert RelationshipType.FRIEND.value == "friend"
        assert RelationshipType.COLLEAGUE.value == "colleague"
        assert RelationshipType.CLIENT.value == "client"
        assert RelationshipType.VENDOR.value == "vendor"
        assert RelationshipType.MENTOR.value == "mentor"
        assert RelationshipType.MENTEE.value == "mentee"
        assert RelationshipType.ACQUAINTANCE.value == "acquaintance"
        assert RelationshipType.OTHER.value == "other"

    def test_count(self):
        """All types accounted for."""
        assert len(list(RelationshipType)) == 9


class TestEventTypeEnum:
    """Tests for EventType enum."""

    def test_values(self):
        """All event types exist."""
        assert EventType.MEETING.value == "meeting"
        assert EventType.APPOINTMENT.value == "appointment"
        assert EventType.DEADLINE.value == "deadline"
        assert EventType.REMINDER.value == "reminder"
        assert EventType.BIRTHDAY.value == "birthday"
        assert EventType.ANNIVERSARY.value == "anniversary"
        assert EventType.HOLIDAY.value == "holiday"
        assert EventType.TASK.value == "task"
        assert EventType.FOCUS_TIME.value == "focus_time"
        assert EventType.TRAVEL.value == "travel"
        assert EventType.OTHER.value == "other"


class TestEventStatusEnum:
    """Tests for EventStatus enum."""

    def test_values(self):
        """All event statuses exist."""
        assert EventStatus.CONFIRMED.value == "confirmed"
        assert EventStatus.TENTATIVE.value == "tentative"
        assert EventStatus.CANCELLED.value == "cancelled"


class TestEmailImportanceEnum:
    """Tests for EmailImportance enum."""

    def test_values(self):
        """All importance levels exist."""
        assert EmailImportance.HIGH.value == "high"
        assert EmailImportance.NORMAL.value == "normal"
        assert EmailImportance.LOW.value == "low"


class TestEmailCategoryEnum:
    """Tests for EmailCategory enum."""

    def test_values(self):
        """All categories exist."""
        assert EmailCategory.PRIMARY.value == "primary"
        assert EmailCategory.SOCIAL.value == "social"
        assert EmailCategory.PROMOTIONS.value == "promotions"
        assert EmailCategory.UPDATES.value == "updates"
        assert EmailCategory.FORUMS.value == "forums"
        assert EmailCategory.SPAM.value == "spam"


class TestTaskPriorityEnum:
    """Tests for TaskPriority enum."""

    def test_values(self):
        """All priorities exist."""
        assert TaskPriority.URGENT.value == "urgent"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.LOW.value == "low"
        assert TaskPriority.SOMEDAY.value == "someday"


class TestTaskStatusEnum:
    """Tests for TaskStatus enum."""

    def test_values(self):
        """All statuses exist."""
        assert TaskStatus.TODO.value == "todo"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.WAITING.value == "waiting"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestContactInfo:
    """Tests for ContactInfo model."""

    def test_basic_creation(self):
        """Create basic contact info."""
        info = ContactInfo(type="email", value="test@example.com")
        assert info.type == "email"
        assert info.value == "test@example.com"
        assert info.label is None
        assert info.primary is False

    def test_with_all_fields(self):
        """Create contact info with all fields."""
        info = ContactInfo(
            type="phone",
            value="+1-555-123-4567",
            label="work",
            primary=True,
        )
        assert info.label == "work"
        assert info.primary is True


class TestContact:
    """Tests for Contact model."""

    def test_basic_creation(self):
        """Create basic contact."""
        contact = Contact(name="John Doe")
        assert contact.name == "John Doe"
        assert contact.contact_type == ContactType.PERSON
        assert contact.relationship == RelationshipType.OTHER
        assert contact.id is not None

    def test_with_emails(self):
        """Create contact with email addresses."""
        contact = Contact(
            name="Jane Doe",
            emails=[
                ContactInfo(type="email", value="jane@work.com", label="work", primary=True),
                ContactInfo(type="email", value="jane@home.com", label="home"),
            ],
        )
        assert len(contact.emails) == 2

    def test_primary_email_with_primary_flag(self):
        """Get primary email when flagged."""
        contact = Contact(
            name="Test",
            emails=[
                ContactInfo(type="email", value="secondary@test.com"),
                ContactInfo(type="email", value="primary@test.com", primary=True),
            ],
        )
        assert contact.primary_email() == "primary@test.com"

    def test_primary_email_first_fallback(self):
        """Get first email when no primary flagged."""
        contact = Contact(
            name="Test",
            emails=[
                ContactInfo(type="email", value="first@test.com"),
                ContactInfo(type="email", value="second@test.com"),
            ],
        )
        assert contact.primary_email() == "first@test.com"

    def test_primary_email_none(self):
        """Return None when no emails."""
        contact = Contact(name="Test")
        assert contact.primary_email() is None

    def test_primary_phone_with_primary_flag(self):
        """Get primary phone when flagged."""
        contact = Contact(
            name="Test",
            phones=[
                ContactInfo(type="phone", value="555-0001"),
                ContactInfo(type="phone", value="555-0002", primary=True),
            ],
        )
        assert contact.primary_phone() == "555-0002"

    def test_primary_phone_first_fallback(self):
        """Get first phone when no primary flagged."""
        contact = Contact(
            name="Test",
            phones=[
                ContactInfo(type="phone", value="555-0001"),
            ],
        )
        assert contact.primary_phone() == "555-0001"

    def test_primary_phone_none(self):
        """Return None when no phones."""
        contact = Contact(name="Test")
        assert contact.primary_phone() is None

    def test_summary_for_embedding_basic(self):
        """Generate basic embedding summary."""
        contact = Contact(name="John Doe")
        summary = contact.summary_for_embedding()
        assert "Contact: John Doe" in summary
        assert "Type: person" in summary

    def test_summary_for_embedding_full(self):
        """Generate full embedding summary."""
        contact = Contact(
            name="Jane Smith",
            organization="Acme Corp",
            job_title="Engineer",
            relationship=RelationshipType.COLLEAGUE,
            notes="Works on AI projects",
            tags=["ai", "ml"],
        )
        summary = contact.summary_for_embedding()
        assert "Organization: Acme Corp" in summary
        assert "Role: Engineer" in summary
        assert "Relationship: colleague" in summary
        assert "Notes: Works on AI projects" in summary
        assert "Tags: ai, ml" in summary


class TestEventAttendee:
    """Tests for EventAttendee model."""

    def test_basic_creation(self):
        """Create basic attendee."""
        attendee = EventAttendee(email="test@example.com")
        assert attendee.email == "test@example.com"
        assert attendee.response == "needs_action"
        assert attendee.organizer is False

    def test_with_all_fields(self):
        """Create attendee with all fields."""
        attendee = EventAttendee(
            email="test@example.com",
            name="Test User",
            response="accepted",
            organizer=True,
            contact_id=uuid4(),
        )
        assert attendee.name == "Test User"
        assert attendee.response == "accepted"
        assert attendee.organizer is True


class TestEventReminder:
    """Tests for EventReminder model."""

    def test_basic_creation(self):
        """Create basic reminder."""
        reminder = EventReminder()
        assert reminder.method == "popup"
        assert reminder.minutes_before == 15

    def test_custom_reminder(self):
        """Create custom reminder."""
        reminder = EventReminder(method="email", minutes_before=60)
        assert reminder.method == "email"
        assert reminder.minutes_before == 60


class TestCalendarEvent:
    """Tests for CalendarEvent model."""

    @pytest.fixture
    def future_event(self):
        """Create a future event."""
        start = datetime.now() + timedelta(hours=2)
        end = start + timedelta(hours=1)
        return CalendarEvent(
            title="Team Meeting",
            start=start,
            end=end,
        )

    @pytest.fixture
    def past_event(self):
        """Create a past event."""
        start = datetime.now() - timedelta(hours=5)
        end = start + timedelta(hours=1)
        return CalendarEvent(
            title="Old Meeting",
            start=start,
            end=end,
        )

    def test_basic_creation(self):
        """Create basic event."""
        start = datetime.now()
        end = start + timedelta(hours=1)
        event = CalendarEvent(title="Test", start=start, end=end)
        assert event.title == "Test"
        assert event.event_type == EventType.MEETING
        assert event.status == EventStatus.CONFIRMED

    def test_duration_property(self, future_event):
        """Duration is calculated correctly."""
        assert future_event.duration == timedelta(hours=1)

    def test_is_past_future_event(self, future_event):
        """Future event is not past."""
        assert future_event.is_past is False

    def test_is_past_past_event(self, past_event):
        """Past event is past."""
        assert past_event.is_past is True

    def test_is_upcoming_within_24h(self):
        """Event starting soon is upcoming."""
        start = datetime.now() + timedelta(hours=12)
        end = start + timedelta(hours=1)
        event = CalendarEvent(title="Soon", start=start, end=end)
        assert event.is_upcoming is True

    def test_is_upcoming_beyond_24h(self):
        """Event starting later is not upcoming."""
        start = datetime.now() + timedelta(days=2)
        end = start + timedelta(hours=1)
        event = CalendarEvent(title="Later", start=start, end=end)
        assert event.is_upcoming is False

    def test_summary_for_embedding_basic(self, future_event):
        """Generate basic embedding summary."""
        summary = future_event.summary_for_embedding()
        assert "Event: Team Meeting" in summary
        assert "Type: meeting" in summary
        assert "When:" in summary

    def test_summary_for_embedding_full(self):
        """Generate full embedding summary."""
        start = datetime.now()
        end = start + timedelta(hours=1)
        event = CalendarEvent(
            title="Project Review",
            start=start,
            end=end,
            description="Quarterly review",
            location="Conference Room A",
            attendees=[
                EventAttendee(email="alice@test.com", name="Alice"),
                EventAttendee(email="bob@test.com", name="Bob"),
            ],
            project="Q4 Review",
        )
        summary = event.summary_for_embedding()
        assert "Description: Quarterly review" in summary
        assert "Location: Conference Room A" in summary
        assert "Attendees: Alice, Bob" in summary
        assert "Project: Q4 Review" in summary


class TestEmailAddress:
    """Tests for EmailAddress model."""

    def test_basic_creation(self):
        """Create basic email address."""
        addr = EmailAddress(email="test@example.com")
        assert addr.email == "test@example.com"
        assert addr.name is None

    def test_with_name(self):
        """Create email address with name."""
        addr = EmailAddress(email="test@example.com", name="Test User")
        assert addr.name == "Test User"


class TestEmailAttachment:
    """Tests for EmailAttachment model."""

    def test_basic_creation(self):
        """Create basic attachment."""
        attachment = EmailAttachment(
            filename="document.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
        )
        assert attachment.filename == "document.pdf"
        assert attachment.size_bytes == 1024


class TestEmail:
    """Tests for Email model."""

    @pytest.fixture
    def basic_email(self):
        """Create basic email."""
        return Email(
            from_address=EmailAddress(email="sender@test.com", name="Sender"),
            to_addresses=[EmailAddress(email="recipient@test.com")],
            subject="Test Subject",
            date=datetime.now(),
        )

    def test_basic_creation(self, basic_email):
        """Create basic email."""
        assert basic_email.subject == "Test Subject"
        assert basic_email.importance == EmailImportance.NORMAL
        assert basic_email.category == EmailCategory.PRIMARY

    def test_summary_for_embedding_basic(self, basic_email):
        """Generate basic embedding summary."""
        summary = basic_email.summary_for_embedding()
        assert "Email from Sender" in summary
        assert "Subject: Test Subject" in summary
        assert "Date:" in summary

    def test_summary_for_embedding_full(self):
        """Generate full embedding summary."""
        email = Email(
            from_address=EmailAddress(email="boss@company.com", name="Boss"),
            to_addresses=[
                EmailAddress(email="me@company.com", name="Me"),
                EmailAddress(email="team@company.com", name="Team"),
            ],
            subject="Important Update",
            snippet="Please review the attached document...",
            date=datetime.now(),
            labels=["important", "work"],
        )
        summary = email.summary_for_embedding()
        assert "Email from Boss" in summary
        assert "Preview: Please review" in summary
        assert "To: Me, Team" in summary
        assert "Labels: important, work" in summary


class TestTask:
    """Tests for Task model."""

    def test_basic_creation(self):
        """Create basic task."""
        task = Task(title="Complete testing")
        assert task.title == "Complete testing"
        assert task.status == TaskStatus.TODO
        assert task.priority == TaskPriority.MEDIUM

    def test_is_overdue_no_due_date(self):
        """Task without due date is not overdue."""
        task = Task(title="Test")
        assert task.is_overdue is False

    def test_is_overdue_completed(self):
        """Completed task is not overdue."""
        task = Task(
            title="Test",
            status=TaskStatus.COMPLETED,
            due_date=datetime.now() - timedelta(days=1),
        )
        assert task.is_overdue is False

    def test_is_overdue_past_due(self):
        """Task past due date is overdue."""
        task = Task(
            title="Test",
            due_date=datetime.now() - timedelta(days=1),
        )
        assert task.is_overdue is True

    def test_is_overdue_future_due(self):
        """Task with future due date is not overdue."""
        task = Task(
            title="Test",
            due_date=datetime.now() + timedelta(days=1),
        )
        assert task.is_overdue is False

    def test_days_until_due_no_date(self):
        """Days until due is None without date."""
        task = Task(title="Test")
        assert task.days_until_due is None

    def test_days_until_due_positive(self):
        """Days until due is positive for future tasks."""
        task = Task(
            title="Test",
            due_date=datetime.now() + timedelta(days=5),
        )
        assert task.days_until_due in [4, 5]  # Could be 4 or 5 depending on time

    def test_days_until_due_negative(self):
        """Days until due is negative for overdue tasks."""
        task = Task(
            title="Test",
            due_date=datetime.now() - timedelta(days=3),
        )
        assert task.days_until_due in [-3, -4]  # Could vary

    def test_summary_for_embedding_basic(self):
        """Generate basic embedding summary."""
        task = Task(title="Write tests")
        summary = task.summary_for_embedding()
        assert "Task: Write tests" in summary
        assert "Status: todo" in summary
        assert "Priority: medium" in summary

    def test_summary_for_embedding_full(self):
        """Generate full embedding summary."""
        task = Task(
            title="Review PR",
            description="Review the pull request for feature X",
            due_date=datetime.now() + timedelta(days=1),
            project="World Weaver",
            tags=["code-review", "urgent"],
        )
        summary = task.summary_for_embedding()
        assert "Description: Review the pull request" in summary
        assert "Due:" in summary
        assert "Project: World Weaver" in summary
        assert "Tags: code-review, urgent" in summary


class TestLocation:
    """Tests for Location model."""

    def test_basic_creation(self):
        """Create basic location."""
        location = Location(name="Office")
        assert location.name == "Office"
        assert location.place_type == "other"

    def test_full_address_complete(self):
        """Full address with all components."""
        location = Location(
            name="HQ",
            address="123 Main St",
            city="San Francisco",
            state="CA",
            postal_code="94102",
            country="USA",
        )
        assert location.full_address() == "123 Main St, San Francisco, CA, 94102, USA"

    def test_full_address_partial(self):
        """Full address with partial components."""
        location = Location(
            name="Office",
            city="New York",
            state="NY",
        )
        assert location.full_address() == "New York, NY"

    def test_full_address_empty(self):
        """Full address with no components."""
        location = Location(name="Unknown")
        assert location.full_address() == ""

    def test_summary_for_embedding_basic(self):
        """Generate basic embedding summary."""
        location = Location(name="Home", place_type="home")
        summary = location.summary_for_embedding()
        assert "Location: Home" in summary
        assert "Type: home" in summary

    def test_summary_for_embedding_full(self):
        """Generate full embedding summary."""
        location = Location(
            name="Work Office",
            place_type="work",
            address="456 Tech Blvd",
            city="Austin",
            state="TX",
            notes="Main development hub",
        )
        summary = location.summary_for_embedding()
        assert "Location: Work Office" in summary
        assert "Address:" in summary
        assert "Notes: Main development hub" in summary


class TestPersonalContext:
    """Tests for PersonalContext dataclass."""

    def test_default_creation(self):
        """Create default context."""
        ctx = PersonalContext()
        assert ctx.timezone == "UTC"
        assert ctx.upcoming_events == []
        assert ctx.overdue_tasks == []

    def test_summary_empty(self):
        """Summary with no items."""
        ctx = PersonalContext()
        assert ctx.summary() == "No pressing items"

    def test_summary_with_events(self):
        """Summary with upcoming events."""
        start = datetime.now() + timedelta(hours=2)
        end = start + timedelta(hours=1)
        ctx = PersonalContext(
            upcoming_events=[
                CalendarEvent(title="Meeting 1", start=start, end=end),
                CalendarEvent(title="Meeting 2", start=start, end=end),
            ]
        )
        summary = ctx.summary()
        assert "Upcoming: 2 events" in summary

    def test_summary_with_overdue_tasks(self):
        """Summary with overdue tasks."""
        ctx = PersonalContext(
            overdue_tasks=[
                Task(title="Task 1", due_date=datetime.now() - timedelta(days=1)),
                Task(title="Task 2", due_date=datetime.now() - timedelta(days=2)),
            ]
        )
        summary = ctx.summary()
        assert "Overdue: 2 tasks" in summary

    def test_summary_with_due_today(self):
        """Summary with tasks due today."""
        ctx = PersonalContext(
            due_today=[Task(title="Today Task")]
        )
        summary = ctx.summary()
        assert "Due today: 1 tasks" in summary

    def test_summary_with_unread_emails(self):
        """Summary with unread important emails."""
        ctx = PersonalContext(
            unread_important_emails=[
                Email(
                    from_address=EmailAddress(email="test@test.com"),
                    subject="Important",
                    date=datetime.now(),
                )
            ]
        )
        summary = ctx.summary()
        assert "Unread important: 1 emails" in summary

    def test_summary_with_birthdays(self):
        """Summary with upcoming birthdays."""
        ctx = PersonalContext(
            upcoming_birthdays=[
                (Contact(name="Alice"), date.today() + timedelta(days=3)),
                (Contact(name="Bob"), date.today() + timedelta(days=5)),
            ]
        )
        summary = ctx.summary()
        assert "Birthdays soon: Alice, Bob" in summary

    def test_summary_full(self):
        """Summary with all items."""
        start = datetime.now() + timedelta(hours=2)
        end = start + timedelta(hours=1)
        ctx = PersonalContext(
            upcoming_events=[CalendarEvent(title="Meeting", start=start, end=end)],
            overdue_tasks=[Task(title="Overdue")],
            due_today=[Task(title="Today")],
            unread_important_emails=[
                Email(
                    from_address=EmailAddress(email="test@test.com"),
                    subject="Test",
                    date=datetime.now(),
                )
            ],
            upcoming_birthdays=[(Contact(name="Alice"), date.today())],
        )
        summary = ctx.summary()
        assert "Upcoming:" in summary
        assert "Overdue:" in summary
        assert "Due today:" in summary
        assert "Unread important:" in summary
        assert "Birthdays soon:" in summary
