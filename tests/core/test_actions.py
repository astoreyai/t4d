"""Tests for core actions module."""

import pytest
from datetime import datetime
from uuid import uuid4, UUID

from t4dm.core.actions import (
    ActionCategory,
    PermissionLevel,
    RiskLevel,
    ActionDefinition,
    ActionStatus,
    ActionRequest,
    ActionResult,
    ActionRegistry,
    ActionExecutor,
    BUILTIN_ACTIONS,
)


class TestActionCategory:
    """Tests for ActionCategory enum."""

    def test_communication_categories(self):
        """Test communication-related categories."""
        assert ActionCategory.EMAIL.value == "email"
        assert ActionCategory.SMS.value == "sms"
        assert ActionCategory.CALL.value == "call"
        assert ActionCategory.CHAT.value == "chat"

    def test_calendar_categories(self):
        """Test calendar-related categories."""
        assert ActionCategory.CALENDAR.value == "calendar"
        assert ActionCategory.REMINDER.value == "reminder"
        assert ActionCategory.TIMER.value == "timer"

    def test_productivity_categories(self):
        """Test productivity categories."""
        assert ActionCategory.TASK.value == "task"
        assert ActionCategory.NOTE.value == "note"
        assert ActionCategory.FOCUS.value == "focus"

    def test_information_categories(self):
        """Test information categories."""
        assert ActionCategory.SEARCH.value == "search"
        assert ActionCategory.LOOKUP.value == "lookup"
        assert ActionCategory.CALCULATE.value == "calculate"

    def test_file_categories(self):
        """Test file categories."""
        assert ActionCategory.FILE.value == "file"
        assert ActionCategory.DOCUMENT.value == "document"

    def test_financial_categories(self):
        """Test financial categories."""
        assert ActionCategory.FINANCE.value == "finance"
        assert ActionCategory.PAYMENT.value == "payment"

    def test_smart_home_categories(self):
        """Test smart home categories."""
        assert ActionCategory.HOME.value == "home"
        assert ActionCategory.DEVICE.value == "device"

    def test_system_categories(self):
        """Test system categories."""
        assert ActionCategory.SYSTEM.value == "system"
        assert ActionCategory.CODE.value == "code"
        assert ActionCategory.AUTOMATION.value == "automation"

    def test_other_categories(self):
        """Test remaining categories."""
        assert ActionCategory.SOCIAL.value == "social"
        assert ActionCategory.NAVIGATION.value == "navigation"


class TestPermissionLevel:
    """Tests for PermissionLevel enum."""

    def test_permission_values(self):
        """Test permission level values."""
        assert PermissionLevel.ALLOWED.value == "allowed"
        assert PermissionLevel.LOGGED.value == "logged"
        assert PermissionLevel.CONFIRM.value == "confirm"
        assert PermissionLevel.VERIFY.value == "verify"
        assert PermissionLevel.BLOCKED.value == "blocked"

    def test_permission_ordering(self):
        """Test that all permission levels are defined."""
        levels = list(PermissionLevel)
        assert len(levels) == 5


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_values(self):
        """Test risk level values."""
        assert RiskLevel.NONE.value == "none"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_risk_count(self):
        """Test all risk levels exist."""
        levels = list(RiskLevel)
        assert len(levels) == 5


class TestActionDefinition:
    """Tests for ActionDefinition dataclass."""

    def test_basic_definition(self):
        """Test basic action definition."""
        action = ActionDefinition(
            name="test.action",
            category=ActionCategory.SYSTEM,
            description="Test action",
            risk_level=RiskLevel.LOW,
            default_permission=PermissionLevel.LOGGED,
        )
        assert action.name == "test.action"
        assert action.category == ActionCategory.SYSTEM
        assert action.risk_level == RiskLevel.LOW
        assert action.default_permission == PermissionLevel.LOGGED

    def test_definition_with_handler(self):
        """Test definition with handler."""
        action = ActionDefinition(
            name="test.with_handler",
            category=ActionCategory.EMAIL,
            description="Action with handler",
            risk_level=RiskLevel.MEDIUM,
            default_permission=PermissionLevel.CONFIRM,
            handler="mcp__test__handler",
        )
        assert action.handler == "mcp__test__handler"

    def test_definition_with_parameters(self):
        """Test definition with parameters."""
        action = ActionDefinition(
            name="test.params",
            category=ActionCategory.EMAIL,
            description="Action with params",
            risk_level=RiskLevel.LOW,
            default_permission=PermissionLevel.LOGGED,
            parameters={"to": {"type": "str", "required": True}},
        )
        assert "to" in action.parameters
        assert action.parameters["to"]["required"] is True

    def test_definition_with_constraints(self):
        """Test definition with constraints."""
        action = ActionDefinition(
            name="test.constraints",
            category=ActionCategory.PAYMENT,
            description="Action with constraints",
            risk_level=RiskLevel.CRITICAL,
            default_permission=PermissionLevel.VERIFY,
            requires_context=["verified_identity"],
            cooldown_seconds=60,
            daily_limit=10,
            rate_limit=(5, 60),
        )
        assert "verified_identity" in action.requires_context
        assert action.cooldown_seconds == 60
        assert action.daily_limit == 10
        assert action.rate_limit == (5, 60)

    def test_definition_with_confirmation(self):
        """Test definition with confirmation template."""
        action = ActionDefinition(
            name="test.confirm",
            category=ActionCategory.FILE,
            description="Action with confirmation",
            risk_level=RiskLevel.HIGH,
            default_permission=PermissionLevel.CONFIRM,
            confirmation_template="Delete file '{filename}'?",
            undo_available=True,
        )
        assert action.confirmation_template == "Delete file '{filename}'?"
        assert action.undo_available is True

    def test_default_values(self):
        """Test default values are set."""
        action = ActionDefinition(
            name="test.defaults",
            category=ActionCategory.LOOKUP,
            description="Test defaults",
            risk_level=RiskLevel.NONE,
            default_permission=PermissionLevel.ALLOWED,
        )
        assert action.handler is None
        assert action.parameters == {}
        assert action.requires_context == []
        assert action.cooldown_seconds == 0
        assert action.daily_limit is None
        assert action.rate_limit is None
        assert action.confirmation_template is None
        assert action.undo_available is False


class TestActionStatus:
    """Tests for ActionStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert ActionStatus.PENDING.value == "pending"
        assert ActionStatus.AWAITING_CONFIRMATION.value == "awaiting_confirmation"
        assert ActionStatus.EXECUTING.value == "executing"
        assert ActionStatus.COMPLETED.value == "completed"
        assert ActionStatus.FAILED.value == "failed"
        assert ActionStatus.CANCELLED.value == "cancelled"
        assert ActionStatus.BLOCKED.value == "blocked"

    def test_status_count(self):
        """All statuses are defined."""
        assert len(list(ActionStatus)) == 7


class TestActionRequest:
    """Tests for ActionRequest dataclass."""

    def test_basic_request(self):
        """Test basic request creation."""
        request = ActionRequest(
            action_name="email.send",
            category=ActionCategory.EMAIL,
            parameters={"to": ["test@example.com"]},
        )
        assert request.action_name == "email.send"
        assert request.category == ActionCategory.EMAIL
        assert request.parameters["to"] == ["test@example.com"]

    def test_request_defaults(self):
        """Test request default values."""
        request = ActionRequest()
        assert request.action_name == ""
        assert request.category == ActionCategory.SYSTEM
        assert request.parameters == {}
        assert request.status == ActionStatus.PENDING
        assert isinstance(request.id, UUID)
        assert isinstance(request.created_at, datetime)

    def test_request_with_context(self):
        """Test request with context."""
        request = ActionRequest(
            action_name="calendar.create",
            category=ActionCategory.CALENDAR,
            session_id="session-123",
            user_utterance="Schedule a meeting tomorrow",
            extracted_intent="create_event",
        )
        assert request.session_id == "session-123"
        assert request.user_utterance == "Schedule a meeting tomorrow"
        assert request.extracted_intent == "create_event"

    def test_request_confirmation_fields(self):
        """Test confirmation-related fields."""
        request = ActionRequest(
            action_name="file.delete",
            requires_confirmation=True,
            confirmation_prompt="Delete file 'test.txt'?",
        )
        assert request.requires_confirmation is True
        assert request.confirmation_prompt == "Delete file 'test.txt'?"
        assert request.confirmed is False
        assert request.confirmed_at is None


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ActionResult(
            success=True,
            action_id=uuid4(),
            action_name="email.send",
            data={"message_id": "123"},
            message="Email sent successfully",
        )
        assert result.success is True
        assert result.data == {"message_id": "123"}
        assert result.message == "Email sent successfully"

    def test_failure_result(self):
        """Test failure result."""
        result = ActionResult(
            success=False,
            action_id=uuid4(),
            action_name="payment.send",
            message="Insufficient funds",
        )
        assert result.success is False
        assert result.message == "Insufficient funds"

    def test_result_with_voice(self):
        """Test result with spoken response."""
        result = ActionResult(
            success=True,
            action_id=uuid4(),
            action_name="reminder.set",
            spoken_response="I've set a reminder for 3pm.",
        )
        assert result.spoken_response == "I've set a reminder for 3pm."

    def test_result_with_undo(self):
        """Test result with undo capability."""
        result = ActionResult(
            success=True,
            action_id=uuid4(),
            action_name="task.complete",
            undo_available=True,
            undo_token="undo-abc123",
        )
        assert result.undo_available is True
        assert result.undo_token == "undo-abc123"

    def test_result_with_followup(self):
        """Test result with follow-up actions."""
        result = ActionResult(
            success=True,
            action_id=uuid4(),
            action_name="calendar.create",
            follow_up_actions=["reminder.set", "email.send"],
            related_entities=[uuid4(), uuid4()],
        )
        assert len(result.follow_up_actions) == 2
        assert len(result.related_entities) == 2


class TestBuiltinActions:
    """Tests for BUILTIN_ACTIONS dictionary."""

    def test_email_actions_exist(self):
        """Test email actions are defined."""
        email_actions = ["email.send", "email.reply", "email.forward",
                        "email.archive", "email.delete", "email.search", "email.read"]
        for action in email_actions:
            assert action in BUILTIN_ACTIONS

    def test_calendar_actions_exist(self):
        """Test calendar actions are defined."""
        calendar_actions = ["calendar.create", "calendar.update", "calendar.delete",
                           "calendar.rsvp", "calendar.list", "calendar.find_time"]
        for action in calendar_actions:
            assert action in BUILTIN_ACTIONS

    def test_task_actions_exist(self):
        """Test task actions are defined."""
        task_actions = ["task.create", "task.complete", "task.update",
                        "task.delete", "task.list", "task.prioritize"]
        for action in task_actions:
            assert action in BUILTIN_ACTIONS

    def test_file_actions_exist(self):
        """Test file actions are defined."""
        file_actions = ["file.read", "file.create", "file.update",
                        "file.delete", "file.share", "document.create"]
        for action in file_actions:
            assert action in BUILTIN_ACTIONS

    def test_memory_actions_exist(self):
        """Test memory actions are defined."""
        memory_actions = ["memory.store", "memory.recall", "memory.forget"]
        for action in memory_actions:
            assert action in BUILTIN_ACTIONS

    def test_high_risk_actions_require_confirmation(self):
        """Test high-risk actions require confirmation."""
        high_risk_actions = ["email.send", "payment.send", "file.delete", "social.post"]
        for action_name in high_risk_actions:
            action = BUILTIN_ACTIONS.get(action_name)
            if action:
                assert action.default_permission in (
                    PermissionLevel.CONFIRM, PermissionLevel.VERIFY
                )

    def test_read_actions_are_allowed(self):
        """Test read-only actions are allowed without confirmation."""
        read_actions = ["email.read", "email.search", "calendar.list",
                        "task.list", "file.read", "memory.recall"]
        for action_name in read_actions:
            action = BUILTIN_ACTIONS.get(action_name)
            if action:
                assert action.default_permission == PermissionLevel.ALLOWED


class TestActionRegistry:
    """Tests for ActionRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return ActionRegistry()

    def test_initialization_loads_builtins(self, registry):
        """Test registry loads built-in actions."""
        assert len(registry.list_all()) > 0
        assert registry.get("email.send") is not None

    def test_register_custom_action(self, registry):
        """Test registering a custom action."""
        custom = ActionDefinition(
            name="custom.action",
            category=ActionCategory.SYSTEM,
            description="Custom action",
            risk_level=RiskLevel.LOW,
            default_permission=PermissionLevel.LOGGED,
        )
        registry.register(custom)
        assert registry.get("custom.action") is custom

    def test_get_nonexistent_returns_none(self, registry):
        """Test getting nonexistent action."""
        assert registry.get("nonexistent.action") is None

    def test_list_by_category(self, registry):
        """Test listing actions by category."""
        email_actions = registry.list_by_category(ActionCategory.EMAIL)
        assert len(email_actions) > 0
        assert all(a.category == ActionCategory.EMAIL for a in email_actions)

    def test_list_all(self, registry):
        """Test listing all actions."""
        all_actions = registry.list_all()
        assert len(all_actions) >= len(BUILTIN_ACTIONS)

    def test_get_permission_default(self, registry):
        """Test getting default permission."""
        perm = registry.get_permission("email.send")
        assert perm == PermissionLevel.CONFIRM

    def test_get_permission_for_unknown(self, registry):
        """Test permission for unknown action is blocked."""
        perm = registry.get_permission("unknown.action")
        assert perm == PermissionLevel.BLOCKED

    def test_set_permission_override(self, registry):
        """Test setting permission override."""
        registry.set_permission("email.send", PermissionLevel.BLOCKED)
        assert registry.get_permission("email.send") == PermissionLevel.BLOCKED

    def test_set_category_permission(self, registry):
        """Test setting category-wide permission."""
        registry.set_category_permission(ActionCategory.EMAIL, PermissionLevel.BLOCKED)
        # Should affect actions in the category that don't have direct override
        assert registry.get_permission("email.search") == PermissionLevel.BLOCKED

    def test_is_allowed(self, registry):
        """Test is_allowed check."""
        assert registry.is_allowed("email.send") is True
        registry.set_permission("email.send", PermissionLevel.BLOCKED)
        assert registry.is_allowed("email.send") is False

    def test_requires_confirmation(self, registry):
        """Test requires_confirmation check."""
        # CONFIRM actions require confirmation
        assert registry.requires_confirmation("email.send") is True
        # ALLOWED actions don't
        assert registry.requires_confirmation("email.read") is False
        # VERIFY actions do
        assert registry.requires_confirmation("payment.send") is True

    def test_permission_override_takes_precedence(self, registry):
        """Test direct override takes precedence over category."""
        registry.set_category_permission(ActionCategory.EMAIL, PermissionLevel.BLOCKED)
        registry.set_permission("email.send", PermissionLevel.ALLOWED)
        # Direct override should take precedence
        assert registry.get_permission("email.send") == PermissionLevel.ALLOWED


class TestActionExecutor:
    """Tests for ActionExecutor class."""

    @pytest.fixture
    def registry(self):
        """Create registry."""
        return ActionRegistry()

    @pytest.fixture
    def executor(self, registry):
        """Create executor."""
        return ActionExecutor(registry)

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, executor):
        """Test executing unknown action fails."""
        request = ActionRequest(action_name="unknown.action")
        result = await executor.execute(request)
        assert result.success is False
        assert "Unknown action" in result.message

    @pytest.mark.asyncio
    async def test_execute_blocked_action(self, executor, registry):
        """Test executing blocked action fails."""
        registry.set_permission("email.send", PermissionLevel.BLOCKED)
        request = ActionRequest(action_name="email.send")
        result = await executor.execute(request)
        assert result.success is False
        assert request.status == ActionStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_execute_allowed_action(self, executor):
        """Test executing allowed action succeeds."""
        request = ActionRequest(action_name="email.read")
        result = await executor.execute(request)
        assert result.success is True
        assert request.status == ActionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_confirm_action_without_confirmation(self, executor):
        """Test CONFIRM action without confirmation returns prompt."""
        request = ActionRequest(
            action_name="email.send",
            parameters={"to": "test@example.com", "subject": "Test"},
        )
        result = await executor.execute(request)
        assert result.success is False
        assert request.status == ActionStatus.AWAITING_CONFIRMATION
        assert request.confirmation_prompt is not None

    @pytest.mark.asyncio
    async def test_execute_confirm_action_with_confirmation(self, executor):
        """Test CONFIRM action with confirmation succeeds."""
        request = ActionRequest(
            action_name="email.send",
            confirmed=True,
        )
        result = await executor.execute(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_rate_limit_cooldown(self, executor, registry):
        """Test cooldown rate limiting."""
        # Create action with cooldown
        action = ActionDefinition(
            name="test.cooldown",
            category=ActionCategory.SYSTEM,
            description="Test cooldown",
            risk_level=RiskLevel.NONE,
            default_permission=PermissionLevel.ALLOWED,
            cooldown_seconds=3600,  # 1 hour
        )
        registry.register(action)

        # First execution succeeds
        request1 = ActionRequest(action_name="test.cooldown")
        result1 = await executor.execute(request1)
        assert result1.success is True

        # Second execution fails due to cooldown
        request2 = ActionRequest(action_name="test.cooldown")
        result2 = await executor.execute(request2)
        assert result2.success is False
        assert "Rate limit" in result2.message

    @pytest.mark.asyncio
    async def test_rate_limit_daily(self, executor, registry):
        """Test daily limit rate limiting."""
        action = ActionDefinition(
            name="test.daily",
            category=ActionCategory.SYSTEM,
            description="Test daily limit",
            risk_level=RiskLevel.NONE,
            default_permission=PermissionLevel.ALLOWED,
            daily_limit=2,
        )
        registry.register(action)

        # First two executions succeed
        for _ in range(2):
            request = ActionRequest(action_name="test.daily")
            result = await executor.execute(request)
            assert result.success is True

        # Third fails due to daily limit
        request = ActionRequest(action_name="test.daily")
        result = await executor.execute(request)
        assert result.success is False
        assert "Rate limit" in result.message

    def test_confirm_pending_request(self, executor):
        """Test confirming a pending request."""
        # Create a request and add to history manually
        request = ActionRequest(
            action_name="email.send",
            status=ActionStatus.AWAITING_CONFIRMATION,
        )
        executor._history.append(request)

        # Confirm it
        confirmed = executor.confirm(request.id)
        assert confirmed is not None
        assert confirmed.confirmed is True
        assert confirmed.confirmed_at is not None

    def test_confirm_nonexistent_request(self, executor):
        """Test confirming nonexistent request."""
        result = executor.confirm(uuid4())
        assert result is None

    def test_cancel_pending_request(self, executor):
        """Test cancelling a pending request."""
        request = ActionRequest(
            action_name="email.send",
            status=ActionStatus.AWAITING_CONFIRMATION,
        )
        executor._history.append(request)

        result = executor.cancel(request.id)
        assert result is True
        assert request.status == ActionStatus.CANCELLED

    def test_cancel_nonexistent_request(self, executor):
        """Test cancelling nonexistent request."""
        result = executor.cancel(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_get_history(self, executor):
        """Test getting action history."""
        # Execute a few actions
        for _ in range(3):
            request = ActionRequest(action_name="email.read")
            await executor.execute(request)

        history = executor.get_history()
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, executor):
        """Test history with limit."""
        for _ in range(5):
            request = ActionRequest(action_name="email.read")
            await executor.execute(request)

        history = executor.get_history(limit=2)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_history_by_category(self, executor):
        """Test history filtered by category."""
        # Execute email action
        email_req = ActionRequest(
            action_name="email.read",
            category=ActionCategory.EMAIL,
        )
        await executor.execute(email_req)

        # Execute calendar action
        calendar_req = ActionRequest(
            action_name="calendar.list",
            category=ActionCategory.CALENDAR,
        )
        await executor.execute(calendar_req)

        # Filter by EMAIL
        email_history = executor.get_history(category=ActionCategory.EMAIL)
        assert all(r.category == ActionCategory.EMAIL for r in email_history)

    def test_format_confirmation_with_template(self, executor, registry):
        """Test confirmation formatting with template."""
        action = registry.get("email.send")
        prompt = executor._format_confirmation(
            action,
            {"to": "test@example.com", "subject": "Hello"},
        )
        assert "test@example.com" in prompt
        assert "Hello" in prompt

    def test_format_confirmation_without_template(self, executor, registry):
        """Test confirmation formatting without template."""
        action = ActionDefinition(
            name="test.no_template",
            category=ActionCategory.SYSTEM,
            description="No template",
            risk_level=RiskLevel.HIGH,
            default_permission=PermissionLevel.CONFIRM,
        )
        prompt = executor._format_confirmation(action, {})
        assert "test.no_template" in prompt

    def test_format_confirmation_with_missing_params(self, executor, registry):
        """Test confirmation when params don't match template."""
        action = registry.get("email.send")
        # Missing 'to' and 'subject' params
        prompt = executor._format_confirmation(action, {})
        # Should fall back to default
        assert "email.send" in prompt
