"""
Action Framework for T4DM AI Assistant.

Defines all actionable operations the assistant can perform,
with permission levels, confirmation requirements, and audit logging.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Action Categories and Permissions
# =============================================================================

class ActionCategory(str, Enum):
    """Categories of actions."""
    # Communication
    EMAIL = "email"
    SMS = "sms"
    CALL = "call"
    CHAT = "chat"  # Slack, Discord, etc.

    # Calendar & Time
    CALENDAR = "calendar"
    REMINDER = "reminder"
    TIMER = "timer"

    # Tasks & Productivity
    TASK = "task"
    NOTE = "note"
    FOCUS = "focus"

    # Information
    SEARCH = "search"
    LOOKUP = "lookup"
    CALCULATE = "calculate"

    # Files & Documents
    FILE = "file"
    DOCUMENT = "document"

    # Financial
    FINANCE = "finance"
    PAYMENT = "payment"

    # Smart Home / IoT
    HOME = "home"
    DEVICE = "device"

    # Social
    SOCIAL = "social"

    # System
    SYSTEM = "system"
    CODE = "code"
    AUTOMATION = "automation"

    # Navigation
    NAVIGATION = "navigation"


class PermissionLevel(str, Enum):
    """Permission levels for actions."""
    # Always allowed without confirmation
    ALLOWED = "allowed"

    # Allowed but logged
    LOGGED = "logged"

    # Requires verbal/explicit confirmation
    CONFIRM = "confirm"

    # Requires extra verification (e.g., PIN, 2FA)
    VERIFY = "verify"

    # Blocked - cannot perform
    BLOCKED = "blocked"


class RiskLevel(str, Enum):
    """Risk assessment for actions."""
    NONE = "none"          # Reading data
    LOW = "low"            # Creating data
    MEDIUM = "medium"      # Modifying data
    HIGH = "high"          # Deleting data, sending messages
    CRITICAL = "critical"  # Financial, irreversible


# =============================================================================
# Action Definitions
# =============================================================================

@dataclass
class ActionDefinition:
    """Definition of an available action."""
    name: str
    category: ActionCategory
    description: str
    risk_level: RiskLevel
    default_permission: PermissionLevel

    # Execution
    handler: str | None = None  # MCP tool name or function path
    parameters: dict[str, Any] = field(default_factory=dict)

    # Constraints
    requires_context: list[str] = field(default_factory=list)  # Required context
    cooldown_seconds: int = 0  # Min time between invocations
    daily_limit: int | None = None  # Max per day
    rate_limit: tuple[int, int] | None = None  # (count, seconds)

    # Confirmation
    confirmation_template: str | None = None
    undo_available: bool = False


# =============================================================================
# Action Request and Result
# =============================================================================

class ActionStatus(str, Enum):
    """Status of an action request."""
    PENDING = "pending"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class ActionRequest:
    """A request to perform an action."""
    id: UUID = field(default_factory=uuid4)
    action_name: str = ""
    category: ActionCategory = ActionCategory.SYSTEM
    parameters: dict[str, Any] = field(default_factory=dict)

    # Context
    session_id: str = ""
    user_utterance: str | None = None  # Original voice/text
    extracted_intent: str | None = None

    # Status
    status: ActionStatus = ActionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: datetime | None = None

    # Confirmation
    requires_confirmation: bool = False
    confirmation_prompt: str | None = None
    confirmed: bool = False
    confirmed_at: datetime | None = None

    # Result
    result: Any = None
    error: str | None = None


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    action_id: UUID
    action_name: str

    # Output
    data: Any = None
    message: str = ""

    # For voice response
    spoken_response: str | None = None

    # Undo
    undo_available: bool = False
    undo_token: str | None = None

    # Follow-up
    follow_up_actions: list[str] = field(default_factory=list)
    related_entities: list[UUID] = field(default_factory=list)


# =============================================================================
# Built-in Action Definitions
# =============================================================================

BUILTIN_ACTIONS: dict[str, ActionDefinition] = {
    # -------------------------------------------------------------------------
    # Email Actions
    # -------------------------------------------------------------------------
    "email.send": ActionDefinition(
        name="email.send",
        category=ActionCategory.EMAIL,
        description="Send an email to one or more recipients",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        handler="mcp__google-workspace__gmail_send_email",
        parameters={
            "to": {"type": "list[str]", "required": True},
            "subject": {"type": "str", "required": True},
            "body": {"type": "str", "required": True},
            "cc": {"type": "list[str]", "required": False},
        },
        confirmation_template="Send email to {to} with subject '{subject}'?",
    ),
    "email.reply": ActionDefinition(
        name="email.reply",
        category=ActionCategory.EMAIL,
        description="Reply to an email",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        handler="mcp__google-workspace__gmail_send_email",
        confirmation_template="Reply to {from_address} about '{subject}'?",
    ),
    "email.forward": ActionDefinition(
        name="email.forward",
        category=ActionCategory.EMAIL,
        description="Forward an email to someone",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Forward email '{subject}' to {to}?",
    ),
    "email.archive": ActionDefinition(
        name="email.archive",
        category=ActionCategory.EMAIL,
        description="Archive an email",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.LOGGED,
        undo_available=True,
    ),
    "email.delete": ActionDefinition(
        name="email.delete",
        category=ActionCategory.EMAIL,
        description="Delete an email",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
        undo_available=True,
        confirmation_template="Delete email '{subject}'?",
    ),
    "email.search": ActionDefinition(
        name="email.search",
        category=ActionCategory.EMAIL,
        description="Search emails",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
        handler="mcp__google-workspace__gmail_search_emails",
    ),
    "email.read": ActionDefinition(
        name="email.read",
        category=ActionCategory.EMAIL,
        description="Read email content",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
        handler="mcp__google-workspace__gmail_read_email",
    ),

    # -------------------------------------------------------------------------
    # Calendar Actions
    # -------------------------------------------------------------------------
    "calendar.create": ActionDefinition(
        name="calendar.create",
        category=ActionCategory.CALENDAR,
        description="Create a calendar event",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.CONFIRM,
        handler="mcp__google-workspace__calendar_create_event",
        confirmation_template="Create event '{summary}' on {start}?",
        undo_available=True,
    ),
    "calendar.update": ActionDefinition(
        name="calendar.update",
        category=ActionCategory.CALENDAR,
        description="Update a calendar event",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Update event '{summary}'?",
        undo_available=True,
    ),
    "calendar.delete": ActionDefinition(
        name="calendar.delete",
        category=ActionCategory.CALENDAR,
        description="Delete a calendar event",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Delete event '{summary}'?",
    ),
    "calendar.rsvp": ActionDefinition(
        name="calendar.rsvp",
        category=ActionCategory.CALENDAR,
        description="Respond to a calendar invitation",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="RSVP '{response}' to '{summary}'?",
    ),
    "calendar.list": ActionDefinition(
        name="calendar.list",
        category=ActionCategory.CALENDAR,
        description="List calendar events",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
        handler="mcp__google-workspace__calendar_list_events",
    ),
    "calendar.find_time": ActionDefinition(
        name="calendar.find_time",
        category=ActionCategory.CALENDAR,
        description="Find available time slots",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),

    # -------------------------------------------------------------------------
    # Reminder Actions
    # -------------------------------------------------------------------------
    "reminder.set": ActionDefinition(
        name="reminder.set",
        category=ActionCategory.REMINDER,
        description="Set a reminder",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
        confirmation_template="Remind you to '{message}' at {time}?",
    ),
    "reminder.cancel": ActionDefinition(
        name="reminder.cancel",
        category=ActionCategory.REMINDER,
        description="Cancel a reminder",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "reminder.list": ActionDefinition(
        name="reminder.list",
        category=ActionCategory.REMINDER,
        description="List active reminders",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),

    # -------------------------------------------------------------------------
    # Task Actions
    # -------------------------------------------------------------------------
    "task.create": ActionDefinition(
        name="task.create",
        category=ActionCategory.TASK,
        description="Create a new task",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "task.complete": ActionDefinition(
        name="task.complete",
        category=ActionCategory.TASK,
        description="Mark a task as complete",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
        undo_available=True,
    ),
    "task.update": ActionDefinition(
        name="task.update",
        category=ActionCategory.TASK,
        description="Update a task",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "task.delete": ActionDefinition(
        name="task.delete",
        category=ActionCategory.TASK,
        description="Delete a task",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
    ),
    "task.list": ActionDefinition(
        name="task.list",
        category=ActionCategory.TASK,
        description="List tasks",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "task.prioritize": ActionDefinition(
        name="task.prioritize",
        category=ActionCategory.TASK,
        description="Change task priority",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),

    # -------------------------------------------------------------------------
    # Contact Actions
    # -------------------------------------------------------------------------
    "contact.lookup": ActionDefinition(
        name="contact.lookup",
        category=ActionCategory.LOOKUP,
        description="Look up a contact",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "contact.create": ActionDefinition(
        name="contact.create",
        category=ActionCategory.LOOKUP,
        description="Create a new contact",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "contact.update": ActionDefinition(
        name="contact.update",
        category=ActionCategory.LOOKUP,
        description="Update contact information",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),

    # -------------------------------------------------------------------------
    # SMS/Messaging Actions
    # -------------------------------------------------------------------------
    "sms.send": ActionDefinition(
        name="sms.send",
        category=ActionCategory.SMS,
        description="Send a text message",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Send text to {to}: '{message}'?",
    ),

    # -------------------------------------------------------------------------
    # Call Actions
    # -------------------------------------------------------------------------
    "call.initiate": ActionDefinition(
        name="call.initiate",
        category=ActionCategory.CALL,
        description="Start a phone call",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Call {contact}?",
    ),

    # -------------------------------------------------------------------------
    # Chat/Messaging Platform Actions
    # -------------------------------------------------------------------------
    "chat.send": ActionDefinition(
        name="chat.send",
        category=ActionCategory.CHAT,
        description="Send a chat message (Slack, Discord, etc.)",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Send to {channel}: '{message}'?",
    ),

    # -------------------------------------------------------------------------
    # Note Actions
    # -------------------------------------------------------------------------
    "note.create": ActionDefinition(
        name="note.create",
        category=ActionCategory.NOTE,
        description="Create a note",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "note.append": ActionDefinition(
        name="note.append",
        category=ActionCategory.NOTE,
        description="Append to an existing note",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "note.search": ActionDefinition(
        name="note.search",
        category=ActionCategory.NOTE,
        description="Search notes",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),

    # -------------------------------------------------------------------------
    # Focus/Productivity Actions
    # -------------------------------------------------------------------------
    "focus.start": ActionDefinition(
        name="focus.start",
        category=ActionCategory.FOCUS,
        description="Start focus/do-not-disturb mode",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "focus.end": ActionDefinition(
        name="focus.end",
        category=ActionCategory.FOCUS,
        description="End focus mode",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "timer.set": ActionDefinition(
        name="timer.set",
        category=ActionCategory.TIMER,
        description="Set a timer",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "timer.cancel": ActionDefinition(
        name="timer.cancel",
        category=ActionCategory.TIMER,
        description="Cancel a timer",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),

    # -------------------------------------------------------------------------
    # Search/Information Actions
    # -------------------------------------------------------------------------
    "search.web": ActionDefinition(
        name="search.web",
        category=ActionCategory.SEARCH,
        description="Search the web",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "search.files": ActionDefinition(
        name="search.files",
        category=ActionCategory.SEARCH,
        description="Search files and documents",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
        handler="mcp__google-workspace__drive_search_files",
    ),
    "lookup.weather": ActionDefinition(
        name="lookup.weather",
        category=ActionCategory.LOOKUP,
        description="Get weather information",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "lookup.directions": ActionDefinition(
        name="lookup.directions",
        category=ActionCategory.LOOKUP,
        description="Get directions to a location",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "calculate": ActionDefinition(
        name="calculate",
        category=ActionCategory.CALCULATE,
        description="Perform calculations",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),

    # -------------------------------------------------------------------------
    # File/Document Actions
    # -------------------------------------------------------------------------
    "file.read": ActionDefinition(
        name="file.read",
        category=ActionCategory.FILE,
        description="Read a file",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
        handler="mcp__google-workspace__drive_read_file",
    ),
    "file.create": ActionDefinition(
        name="file.create",
        category=ActionCategory.FILE,
        description="Create a new file",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
        handler="mcp__google-workspace__drive_upload_file",
    ),
    "file.update": ActionDefinition(
        name="file.update",
        category=ActionCategory.FILE,
        description="Update a file",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.LOGGED,
        handler="mcp__google-workspace__drive_update_file",
    ),
    "file.delete": ActionDefinition(
        name="file.delete",
        category=ActionCategory.FILE,
        description="Delete a file",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        handler="mcp__google-workspace__drive_delete_file",
        confirmation_template="Delete file '{filename}'?",
    ),
    "file.share": ActionDefinition(
        name="file.share",
        category=ActionCategory.FILE,
        description="Share a file with someone",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
        handler="mcp__google-workspace__drive_share_file",
        confirmation_template="Share '{filename}' with {email}?",
    ),
    "document.create": ActionDefinition(
        name="document.create",
        category=ActionCategory.DOCUMENT,
        description="Create a Google Doc/Sheet/Slides",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
        handler="mcp__google-workspace__drive_create_doc",
    ),

    # -------------------------------------------------------------------------
    # Financial Actions (HIGH RISK)
    # -------------------------------------------------------------------------
    "finance.check_balance": ActionDefinition(
        name="finance.check_balance",
        category=ActionCategory.FINANCE,
        description="Check account balance",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.VERIFY,  # Extra verification
    ),
    "payment.send": ActionDefinition(
        name="payment.send",
        category=ActionCategory.PAYMENT,
        description="Send a payment",
        risk_level=RiskLevel.CRITICAL,
        default_permission=PermissionLevel.VERIFY,
        confirmation_template="Send ${amount} to {recipient}?",
        requires_context=["verified_identity"],
    ),
    "payment.request": ActionDefinition(
        name="payment.request",
        category=ActionCategory.PAYMENT,
        description="Request a payment",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
    ),

    # -------------------------------------------------------------------------
    # Smart Home Actions
    # -------------------------------------------------------------------------
    "home.lights": ActionDefinition(
        name="home.lights",
        category=ActionCategory.HOME,
        description="Control lights",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "home.thermostat": ActionDefinition(
        name="home.thermostat",
        category=ActionCategory.HOME,
        description="Control thermostat",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "home.lock": ActionDefinition(
        name="home.lock",
        category=ActionCategory.HOME,
        description="Lock/unlock doors",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="{action} the {door}?",
    ),
    "home.alarm": ActionDefinition(
        name="home.alarm",
        category=ActionCategory.HOME,
        description="Arm/disarm security system",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.VERIFY,
    ),
    "device.play_music": ActionDefinition(
        name="device.play_music",
        category=ActionCategory.DEVICE,
        description="Play music",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "device.volume": ActionDefinition(
        name="device.volume",
        category=ActionCategory.DEVICE,
        description="Adjust volume",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),

    # -------------------------------------------------------------------------
    # Social Media Actions
    # -------------------------------------------------------------------------
    "social.post": ActionDefinition(
        name="social.post",
        category=ActionCategory.SOCIAL,
        description="Post to social media",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Post to {platform}: '{content}'?",
    ),

    # -------------------------------------------------------------------------
    # System/Code Actions
    # -------------------------------------------------------------------------
    "system.run_command": ActionDefinition(
        name="system.run_command",
        category=ActionCategory.SYSTEM,
        description="Run a system command",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Run: {command}?",
    ),
    "code.execute": ActionDefinition(
        name="code.execute",
        category=ActionCategory.CODE,
        description="Execute code",
        risk_level=RiskLevel.HIGH,
        default_permission=PermissionLevel.CONFIRM,
    ),
    "code.deploy": ActionDefinition(
        name="code.deploy",
        category=ActionCategory.CODE,
        description="Deploy code to production",
        risk_level=RiskLevel.CRITICAL,
        default_permission=PermissionLevel.VERIFY,
        confirmation_template="Deploy {project} to {environment}?",
    ),
    "automation.create": ActionDefinition(
        name="automation.create",
        category=ActionCategory.AUTOMATION,
        description="Create an automation/workflow",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
    ),
    "automation.trigger": ActionDefinition(
        name="automation.trigger",
        category=ActionCategory.AUTOMATION,
        description="Trigger an automation",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.LOGGED,
    ),

    # -------------------------------------------------------------------------
    # Navigation Actions
    # -------------------------------------------------------------------------
    "navigation.start": ActionDefinition(
        name="navigation.start",
        category=ActionCategory.NAVIGATION,
        description="Start navigation to a destination",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "navigation.share_location": ActionDefinition(
        name="navigation.share_location",
        category=ActionCategory.NAVIGATION,
        description="Share current location",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Share your location with {contact}?",
    ),

    # -------------------------------------------------------------------------
    # Memory Actions (T4DM)
    # -------------------------------------------------------------------------
    "memory.store": ActionDefinition(
        name="memory.store",
        category=ActionCategory.SYSTEM,
        description="Store something in memory",
        risk_level=RiskLevel.LOW,
        default_permission=PermissionLevel.LOGGED,
    ),
    "memory.recall": ActionDefinition(
        name="memory.recall",
        category=ActionCategory.SYSTEM,
        description="Recall from memory",
        risk_level=RiskLevel.NONE,
        default_permission=PermissionLevel.ALLOWED,
    ),
    "memory.forget": ActionDefinition(
        name="memory.forget",
        category=ActionCategory.SYSTEM,
        description="Forget/delete a memory",
        risk_level=RiskLevel.MEDIUM,
        default_permission=PermissionLevel.CONFIRM,
        confirmation_template="Forget memory about '{topic}'?",
    ),
}


# =============================================================================
# Action Registry
# =============================================================================

class ActionRegistry:
    """
    Registry of available actions with permission management.
    """

    def __init__(self) -> None:
        self._actions: dict[str, ActionDefinition] = {}
        self._permission_overrides: dict[str, PermissionLevel] = {}
        self._category_permissions: dict[ActionCategory, PermissionLevel] = {}

        # Load built-in actions
        for name, action in BUILTIN_ACTIONS.items():
            self.register(action)

    def register(self, action: ActionDefinition) -> None:
        """Register an action."""
        self._actions[action.name] = action
        logger.debug(f"Registered action: {action.name}")

    def get(self, name: str) -> ActionDefinition | None:
        """Get an action by name."""
        return self._actions.get(name)

    def list_by_category(self, category: ActionCategory) -> list[ActionDefinition]:
        """List actions in a category."""
        return [a for a in self._actions.values() if a.category == category]

    def list_all(self) -> list[ActionDefinition]:
        """List all actions."""
        return list(self._actions.values())

    def get_permission(self, action_name: str) -> PermissionLevel:
        """Get effective permission for an action."""
        # Check direct override first
        if action_name in self._permission_overrides:
            return self._permission_overrides[action_name]

        # Check action definition
        action = self._actions.get(action_name)
        if not action:
            return PermissionLevel.BLOCKED

        # Check category override
        if action.category in self._category_permissions:
            return self._category_permissions[action.category]

        return action.default_permission

    def set_permission(self, action_name: str, permission: PermissionLevel) -> None:
        """Override permission for a specific action."""
        self._permission_overrides[action_name] = permission

    def set_category_permission(
        self,
        category: ActionCategory,
        permission: PermissionLevel,
    ) -> None:
        """Set permission for all actions in a category."""
        self._category_permissions[category] = permission

    def is_allowed(self, action_name: str) -> bool:
        """Check if action is allowed (not blocked)."""
        return self.get_permission(action_name) != PermissionLevel.BLOCKED

    def requires_confirmation(self, action_name: str) -> bool:
        """Check if action requires confirmation."""
        perm = self.get_permission(action_name)
        return perm in (PermissionLevel.CONFIRM, PermissionLevel.VERIFY)


# =============================================================================
# Action Executor
# =============================================================================

class ActionExecutor:
    """
    Executes actions with permission checking, confirmation, and logging.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        confirmation_callback: Callable[[str], Awaitable[bool]] | None = None,
        verification_callback: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self.registry = registry
        self._confirm = confirmation_callback
        self._verify = verification_callback

        # Execution history
        self._history: list[ActionRequest] = []
        self._daily_counts: dict[str, int] = {}
        self._last_execution: dict[str, datetime] = {}

    async def execute(self, request: ActionRequest) -> ActionResult:
        """
        Execute an action request.

        Returns:
            ActionResult with execution outcome
        """
        action = self.registry.get(request.action_name)
        if not action:
            return ActionResult(
                success=False,
                action_id=request.id,
                action_name=request.action_name,
                message=f"Unknown action: {request.action_name}",
            )

        # Check permission
        permission = self.registry.get_permission(request.action_name)
        if permission == PermissionLevel.BLOCKED:
            request.status = ActionStatus.BLOCKED
            return ActionResult(
                success=False,
                action_id=request.id,
                action_name=request.action_name,
                message="Action is blocked by policy",
            )

        # Check rate limits
        if not self._check_limits(action):
            return ActionResult(
                success=False,
                action_id=request.id,
                action_name=request.action_name,
                message="Rate limit exceeded",
            )

        # Handle confirmation
        if permission == PermissionLevel.CONFIRM:
            if not request.confirmed:
                request.status = ActionStatus.AWAITING_CONFIRMATION
                request.confirmation_prompt = self._format_confirmation(
                    action, request.parameters
                )
                return ActionResult(
                    success=False,
                    action_id=request.id,
                    action_name=request.action_name,
                    message=request.confirmation_prompt,
                    spoken_response=request.confirmation_prompt,
                )

        # Handle verification
        if permission == PermissionLevel.VERIFY:
            if self._verify:
                verified = await self._verify(request.action_name)
                if not verified:
                    request.status = ActionStatus.CANCELLED
                    return ActionResult(
                        success=False,
                        action_id=request.id,
                        action_name=request.action_name,
                        message="Verification failed",
                    )

        # Execute
        request.status = ActionStatus.EXECUTING
        try:
            result = await self._do_execute(action, request)
            request.status = ActionStatus.COMPLETED
            request.executed_at = datetime.now()
            self._record_execution(action)
            return result

        except Exception as e:
            request.status = ActionStatus.FAILED
            request.error = str(e)
            logger.error(f"Action failed: {action.name} - {e}")
            return ActionResult(
                success=False,
                action_id=request.id,
                action_name=request.action_name,
                message=f"Execution failed: {e}",
            )

        finally:
            self._history.append(request)

    async def _do_execute(
        self,
        action: ActionDefinition,
        request: ActionRequest,
    ) -> ActionResult:
        """Actually execute the action (override in subclass for real implementation)."""
        # This is a stub - real implementation would call MCP tools
        logger.info(f"Executing: {action.name} with {request.parameters}")

        return ActionResult(
            success=True,
            action_id=request.id,
            action_name=action.name,
            message=f"Executed {action.name}",
            undo_available=action.undo_available,
        )

    def _check_limits(self, action: ActionDefinition) -> bool:
        """Check rate limits and cooldowns."""
        now = datetime.now()

        # Check cooldown
        if action.cooldown_seconds > 0:
            last = self._last_execution.get(action.name)
            if last and (now - last).total_seconds() < action.cooldown_seconds:
                return False

        # Check daily limit
        if action.daily_limit:
            count = self._daily_counts.get(action.name, 0)
            if count >= action.daily_limit:
                return False

        return True

    def _record_execution(self, action: ActionDefinition) -> None:
        """Record execution for rate limiting."""
        self._last_execution[action.name] = datetime.now()
        self._daily_counts[action.name] = self._daily_counts.get(action.name, 0) + 1

    def _format_confirmation(
        self,
        action: ActionDefinition,
        parameters: dict[str, Any],
    ) -> str:
        """Format confirmation prompt."""
        if action.confirmation_template:
            try:
                return action.confirmation_template.format(**parameters)
            except KeyError:
                pass
        return f"Confirm {action.name}?"

    def confirm(self, request_id: UUID) -> ActionRequest | None:
        """Confirm a pending action."""
        for req in reversed(self._history):
            if req.id == request_id and req.status == ActionStatus.AWAITING_CONFIRMATION:
                req.confirmed = True
                req.confirmed_at = datetime.now()
                return req
        return None

    def cancel(self, request_id: UUID) -> bool:
        """Cancel a pending action."""
        for req in reversed(self._history):
            if req.id == request_id and req.status == ActionStatus.AWAITING_CONFIRMATION:
                req.status = ActionStatus.CANCELLED
                return True
        return False

    def get_history(
        self,
        limit: int = 50,
        category: ActionCategory | None = None,
    ) -> list[ActionRequest]:
        """Get action history."""
        history = self._history
        if category:
            history = [r for r in history if r.category == category]
        return history[-limit:]
