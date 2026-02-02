"""
Personal Data Entities for T4DM.

Extends the tripartite memory system with personal assistant data:
- Contacts (people, organizations)
- Calendar Events (meetings, reminders, deadlines)
- Emails (messages, threads, attachments)
- Tasks (todos, projects)
- Locations (places, addresses)

These entities integrate with external services (Google Workspace, etc.)
while maintaining a unified knowledge graph in T4DM.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================

class ContactType(str, Enum):
    """Type of contact."""
    PERSON = "person"
    ORGANIZATION = "organization"
    GROUP = "group"
    SERVICE = "service"  # e.g., support@company.com


class RelationshipType(str, Enum):
    """Relationship between user and contact."""
    FAMILY = "family"
    FRIEND = "friend"
    COLLEAGUE = "colleague"
    CLIENT = "client"
    VENDOR = "vendor"
    MENTOR = "mentor"
    MENTEE = "mentee"
    ACQUAINTANCE = "acquaintance"
    OTHER = "other"


class EventType(str, Enum):
    """Type of calendar event."""
    MEETING = "meeting"
    APPOINTMENT = "appointment"
    DEADLINE = "deadline"
    REMINDER = "reminder"
    BIRTHDAY = "birthday"
    ANNIVERSARY = "anniversary"
    HOLIDAY = "holiday"
    TASK = "task"
    FOCUS_TIME = "focus_time"
    TRAVEL = "travel"
    OTHER = "other"


class EventStatus(str, Enum):
    """Status of calendar event."""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EmailImportance(str, Enum):
    """Email importance level."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class EmailCategory(str, Enum):
    """Email category for classification."""
    PRIMARY = "primary"
    SOCIAL = "social"
    PROMOTIONS = "promotions"
    UPDATES = "updates"
    FORUMS = "forums"
    SPAM = "spam"


class TaskPriority(str, Enum):
    """Task priority level."""
    URGENT = "urgent"      # Do now
    HIGH = "high"          # Do today
    MEDIUM = "medium"      # Do this week
    LOW = "low"            # Do eventually
    SOMEDAY = "someday"    # Maybe


class TaskStatus(str, Enum):
    """Task status."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"      # Blocked on something
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# =============================================================================
# Contact Entity
# =============================================================================

class ContactInfo(BaseModel):
    """Contact information (email, phone, etc.)."""
    type: str  # "email", "phone", "address", "website", "social"
    value: str
    label: str | None = None  # "work", "home", "mobile"
    primary: bool = False


class Contact(BaseModel):
    """
    A person or organization in the user's network.

    Integrates with Google Contacts, LinkedIn, etc.
    """
    id: UUID = Field(default_factory=uuid4)
    external_id: str | None = None  # Google Contact ID, etc.
    source: str = "manual"  # "google", "linkedin", "manual"

    # Identity
    name: str
    display_name: str | None = None
    contact_type: ContactType = ContactType.PERSON

    # Contact info
    emails: list[ContactInfo] = Field(default_factory=list)
    phones: list[ContactInfo] = Field(default_factory=list)
    addresses: list[ContactInfo] = Field(default_factory=list)

    # Relationships
    relationship: RelationshipType = RelationshipType.OTHER
    organization: str | None = None
    job_title: str | None = None

    # Important dates
    birthday: date | None = None
    anniversary: date | None = None
    custom_dates: dict[str, date] = Field(default_factory=dict)

    # Context
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)

    # T4DM integration
    embedding: list[float] | None = None
    entity_id: UUID | None = None  # Link to WW Entity

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_contacted: datetime | None = None
    interaction_count: int = 0

    def primary_email(self) -> str | None:
        """Get primary email address."""
        for info in self.emails:
            if info.primary:
                return info.value
        return self.emails[0].value if self.emails else None

    def primary_phone(self) -> str | None:
        """Get primary phone number."""
        for info in self.phones:
            if info.primary:
                return info.value
        return self.phones[0].value if self.phones else None

    def summary_for_embedding(self) -> str:
        """Generate text summary for embedding."""
        parts = [
            f"Contact: {self.name}",
            f"Type: {self.contact_type.value}",
        ]
        if self.organization:
            parts.append(f"Organization: {self.organization}")
        if self.job_title:
            parts.append(f"Role: {self.job_title}")
        if self.relationship != RelationshipType.OTHER:
            parts.append(f"Relationship: {self.relationship.value}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return " | ".join(parts)


# =============================================================================
# Calendar Event Entity
# =============================================================================

class EventAttendee(BaseModel):
    """An attendee of a calendar event."""
    email: str
    name: str | None = None
    response: str = "needs_action"  # accepted, declined, tentative
    organizer: bool = False
    contact_id: UUID | None = None  # Link to Contact


class EventReminder(BaseModel):
    """Reminder for an event."""
    method: str = "popup"  # popup, email, sms
    minutes_before: int = 15


class CalendarEvent(BaseModel):
    """
    A calendar event (meeting, deadline, reminder, etc.)

    Integrates with Google Calendar, Outlook, etc.
    """
    id: UUID = Field(default_factory=uuid4)
    external_id: str | None = None  # Google Event ID
    calendar_id: str = "primary"
    source: str = "manual"  # "google", "outlook", "manual"

    # Basic info
    title: str
    description: str | None = None
    event_type: EventType = EventType.MEETING
    status: EventStatus = EventStatus.CONFIRMED

    # Timing
    start: datetime
    end: datetime
    all_day: bool = False
    timezone: str = "UTC"

    # Recurrence
    recurring: bool = False
    recurrence_rule: str | None = None  # RRULE format
    recurrence_id: str | None = None  # For recurring event instances

    # Location
    location: str | None = None
    meeting_link: str | None = None  # Zoom, Meet, etc.

    # Participants
    attendees: list[EventAttendee] = Field(default_factory=list)
    organizer_email: str | None = None

    # Reminders
    reminders: list[EventReminder] = Field(default_factory=list)

    # Context
    tags: list[str] = Field(default_factory=list)
    project: str | None = None  # Link to project

    # T4DM integration
    embedding: list[float] | None = None
    entity_id: UUID | None = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def duration(self) -> timedelta:
        """Event duration."""
        return self.end - self.start

    @property
    def is_past(self) -> bool:
        """Check if event is in the past."""
        return self.end < datetime.now()

    @property
    def is_upcoming(self) -> bool:
        """Check if event is within next 24 hours."""
        now = datetime.now()
        return now <= self.start <= now + timedelta(hours=24)

    def summary_for_embedding(self) -> str:
        """Generate text summary for embedding."""
        parts = [
            f"Event: {self.title}",
            f"Type: {self.event_type.value}",
            f"When: {self.start.strftime('%Y-%m-%d %H:%M')}",
        ]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.attendees:
            names = [a.name or a.email for a in self.attendees[:5]]
            parts.append(f"Attendees: {', '.join(names)}")
        if self.project:
            parts.append(f"Project: {self.project}")
        return " | ".join(parts)


# =============================================================================
# Email Entity
# =============================================================================

class EmailAddress(BaseModel):
    """Email address with optional name."""
    email: str
    name: str | None = None
    contact_id: UUID | None = None


class EmailAttachment(BaseModel):
    """Email attachment metadata."""
    filename: str
    mime_type: str
    size_bytes: int
    attachment_id: str | None = None  # For retrieval


class Email(BaseModel):
    """
    An email message.

    Integrates with Gmail, Outlook, etc.
    """
    id: UUID = Field(default_factory=uuid4)
    external_id: str | None = None  # Gmail message ID
    thread_id: str | None = None  # For threading
    source: str = "gmail"

    # Addressing
    from_address: EmailAddress
    to_addresses: list[EmailAddress] = Field(default_factory=list)
    cc_addresses: list[EmailAddress] = Field(default_factory=list)
    bcc_addresses: list[EmailAddress] = Field(default_factory=list)
    reply_to: EmailAddress | None = None

    # Content
    subject: str
    snippet: str | None = None  # Preview text
    body_text: str | None = None  # Plain text body
    body_html: str | None = None  # HTML body

    # Metadata
    date: datetime
    importance: EmailImportance = EmailImportance.NORMAL
    category: EmailCategory = EmailCategory.PRIMARY

    # Status
    is_read: bool = False
    is_starred: bool = False
    is_draft: bool = False
    is_sent: bool = False

    # Labels/folders
    labels: list[str] = Field(default_factory=list)

    # Attachments
    attachments: list[EmailAttachment] = Field(default_factory=list)
    has_attachments: bool = False

    # Threading
    in_reply_to: str | None = None  # Message-ID header
    references: list[str] = Field(default_factory=list)

    # T4DM integration
    embedding: list[float] | None = None
    entity_id: UUID | None = None

    # Derived
    action_required: bool = False  # Needs response
    follow_up_date: dt.date | None = None

    def summary_for_embedding(self) -> str:
        """Generate text summary for embedding."""
        parts = [
            f"Email from {self.from_address.name or self.from_address.email}",
            f"Subject: {self.subject}",
            f"Date: {self.date.strftime('%Y-%m-%d')}",
        ]
        if self.snippet:
            parts.append(f"Preview: {self.snippet[:200]}")
        if self.to_addresses:
            to_names = [a.name or a.email for a in self.to_addresses[:3]]
            parts.append(f"To: {', '.join(to_names)}")
        if self.labels:
            parts.append(f"Labels: {', '.join(self.labels)}")
        return " | ".join(parts)


# =============================================================================
# Task Entity
# =============================================================================

class Task(BaseModel):
    """
    A task or todo item.

    Integrates with Google Tasks, Todoist, etc.
    """
    id: UUID = Field(default_factory=uuid4)
    external_id: str | None = None
    source: str = "manual"

    # Content
    title: str
    description: str | None = None
    notes: str | None = None

    # Status
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    due_date: datetime | None = None
    completed_at: datetime | None = None
    start_date: datetime | None = None

    # Context
    project: str | None = None
    tags: list[str] = Field(default_factory=list)
    context: str | None = None  # @home, @work, @errands

    # Relationships
    parent_id: UUID | None = None  # Subtask of
    blocked_by: list[UUID] = Field(default_factory=list)
    related_event_id: UUID | None = None
    related_email_id: UUID | None = None
    related_contact_ids: list[UUID] = Field(default_factory=list)

    # Recurrence
    recurring: bool = False
    recurrence_rule: str | None = None

    # T4DM integration
    embedding: list[float] | None = None
    entity_id: UUID | None = None

    @property
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if not self.due_date or self.status == TaskStatus.COMPLETED:
            return False
        return datetime.now() > self.due_date

    @property
    def days_until_due(self) -> int | None:
        """Days until due date."""
        if not self.due_date:
            return None
        delta = self.due_date - datetime.now()
        return delta.days

    def summary_for_embedding(self) -> str:
        """Generate text summary for embedding."""
        parts = [
            f"Task: {self.title}",
            f"Status: {self.status.value}",
            f"Priority: {self.priority.value}",
        ]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.due_date:
            parts.append(f"Due: {self.due_date.strftime('%Y-%m-%d')}")
        if self.project:
            parts.append(f"Project: {self.project}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return " | ".join(parts)


# =============================================================================
# Location Entity
# =============================================================================

class Location(BaseModel):
    """
    A physical location or place.

    For addresses, frequently visited places, etc.
    """
    id: UUID = Field(default_factory=uuid4)
    external_id: str | None = None  # Google Place ID

    # Identity
    name: str
    place_type: str = "other"  # home, work, restaurant, etc.

    # Address
    address: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    postal_code: str | None = None

    # Coordinates
    latitude: float | None = None
    longitude: float | None = None

    # Context
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Usage
    visit_count: int = 0
    last_visited: datetime | None = None

    # T4DM integration
    embedding: list[float] | None = None
    entity_id: UUID | None = None

    def full_address(self) -> str:
        """Get full address string."""
        parts = []
        if self.address:
            parts.append(self.address)
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        if self.postal_code:
            parts.append(self.postal_code)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)

    def summary_for_embedding(self) -> str:
        """Generate text summary for embedding."""
        parts = [
            f"Location: {self.name}",
            f"Type: {self.place_type}",
        ]
        addr = self.full_address()
        if addr:
            parts.append(f"Address: {addr}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return " | ".join(parts)


# =============================================================================
# Personal Context (aggregated view)
# =============================================================================

@dataclass
class PersonalContext:
    """
    Aggregated personal context for a point in time.

    Used to provide rich context to the assistant.
    """
    # Time context
    current_time: datetime = field(default_factory=datetime.now)
    timezone: str = "UTC"

    # Upcoming events (next 24h)
    upcoming_events: list[CalendarEvent] = field(default_factory=list)

    # Due tasks
    overdue_tasks: list[Task] = field(default_factory=list)
    due_today: list[Task] = field(default_factory=list)
    due_this_week: list[Task] = field(default_factory=list)

    # Recent emails needing attention
    unread_important_emails: list[Email] = field(default_factory=list)
    action_required_emails: list[Email] = field(default_factory=list)

    # Relevant contacts (for current context)
    relevant_contacts: list[Contact] = field(default_factory=list)

    # Upcoming birthdays/anniversaries (next 7 days)
    upcoming_birthdays: list[tuple[Contact, date]] = field(default_factory=list)
    upcoming_anniversaries: list[tuple[Contact, date]] = field(default_factory=list)

    # Current location (if available)
    current_location: Location | None = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        parts = []

        if self.upcoming_events:
            parts.append(f"Upcoming: {len(self.upcoming_events)} events in next 24h")

        if self.overdue_tasks:
            parts.append(f"Overdue: {len(self.overdue_tasks)} tasks")

        if self.due_today:
            parts.append(f"Due today: {len(self.due_today)} tasks")

        if self.unread_important_emails:
            parts.append(f"Unread important: {len(self.unread_important_emails)} emails")

        if self.upcoming_birthdays:
            names = [c.name for c, _ in self.upcoming_birthdays[:3]]
            parts.append(f"Birthdays soon: {', '.join(names)}")

        return " | ".join(parts) if parts else "No pressing items"
