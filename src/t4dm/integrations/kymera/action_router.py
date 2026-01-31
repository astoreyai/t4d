"""
Voice Action Router - Routes parsed intents to appropriate handlers.

Handles permission checking, confirmation flows, and execution.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from t4dm.core.actions import (
    ActionDefinition,
    ActionRegistry,
    ActionRequest,
    ActionStatus,
    PermissionLevel,
)
from t4dm.integrations.kymera.bridge import VoiceContext, VoiceMemoryBridge

logger = logging.getLogger(__name__)


@dataclass
class VoiceActionResult:
    """Result of voice action execution."""
    success: bool
    spoken_response: str
    data: Any = None
    requires_confirmation: bool = False
    confirmation_prompt: str | None = None
    follow_up_actions: list[str] = None

    def __post_init__(self):
        if self.follow_up_actions is None:
            self.follow_up_actions = []


# Type alias for handler functions
ActionHandler = Callable[[ActionRequest, VoiceContext], Awaitable[VoiceActionResult]]


class VoiceActionRouter:
    """
    Routes voice intents to action handlers with voice-specific features.

    Features:
    - Permission-aware routing
    - Voice confirmation flow
    - Natural language responses
    - Memory integration
    """

    # Phrases for confirmation
    CONFIRM_PHRASES = ["yes", "confirm", "do it", "go ahead", "send it", "yes please", "yeah"]
    CANCEL_PHRASES = ["no", "cancel", "stop", "nevermind", "never mind", "don't"]

    def __init__(
        self,
        registry: ActionRegistry,
        memory_bridge: VoiceMemoryBridge,
        mcp_client: Any,  # MCP client for Google Workspace etc.
        claude_client: Any,  # Claude client for chat fallback
    ):
        """
        Initialize voice action router.

        Args:
            registry: Action registry with definitions
            memory_bridge: Memory bridge for WW operations
            mcp_client: MCP client for external tools
            claude_client: Claude client for chat
        """
        self.registry = registry
        self.memory = memory_bridge
        self.mcp = mcp_client
        self.claude = claude_client

        # Pending confirmations
        self._pending_confirmations: dict[str, ActionRequest] = {}

        # Handler registry
        self._handlers: dict[str, ActionHandler] = {}
        self._register_handlers()

    def _register_handlers(self):
        """Register action handlers."""
        # Email handlers
        self._handlers["email.send"] = self._handle_email_send
        self._handlers["email.read"] = self._handle_email_read
        self._handlers["email.search"] = self._handle_email_search
        self._handlers["email.reply"] = self._handle_email_reply

        # Calendar handlers
        self._handlers["calendar.create"] = self._handle_calendar_create
        self._handlers["calendar.list"] = self._handle_calendar_list
        self._handlers["calendar.query"] = self._handle_calendar_query

        # Reminder handlers
        self._handlers["reminder.set"] = self._handle_reminder_set
        self._handlers["reminder.list"] = self._handle_reminder_list

        # Task handlers
        self._handlers["task.create"] = self._handle_task_create
        self._handlers["task.list"] = self._handle_task_list
        self._handlers["task.complete"] = self._handle_task_complete

        # Memory handlers
        self._handlers["memory.store"] = self._handle_memory_store
        self._handlers["memory.recall"] = self._handle_memory_recall

        # Contact handlers
        self._handlers["contact.lookup"] = self._handle_contact_lookup

        # Lookup handlers
        self._handlers["lookup.time"] = self._handle_lookup_time
        self._handlers["lookup.date"] = self._handle_lookup_date

        # Default: Claude chat
        self._handlers["claude.chat"] = self._handle_claude_chat

    async def route(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """
        Route action request to appropriate handler.

        Args:
            request: Action request to route
            context: Voice context

        Returns:
            VoiceActionResult with spoken response
        """
        action_name = request.action_name

        # Check for pending confirmation response
        if context.session_id in self._pending_confirmations:
            return await self._handle_confirmation_response(request, context)

        # Get action definition
        action_def = self.registry.get(action_name)

        # Check permission
        permission = self.registry.get_permission(action_name)

        if permission == PermissionLevel.BLOCKED:
            return VoiceActionResult(
                success=False,
                spoken_response="Sorry, that action is not allowed.",
            )

        # Check if confirmation needed
        if permission in (PermissionLevel.CONFIRM, PermissionLevel.VERIFY):
            if not request.confirmed:
                return await self._request_confirmation(request, action_def, context)

        # Get handler
        handler = self._handlers.get(action_name)
        if not handler:
            # Try to find a partial match or use default
            prefix = action_name.split(".")[0]
            handler = self._handlers.get(f"{prefix}.default")
            if not handler:
                handler = self._handlers["claude.chat"]

        # Execute handler
        try:
            result = await handler(request, context)
            request.status = ActionStatus.COMPLETED

            # Store successful actions in memory
            if result.success and action_name not in ("claude.chat", "memory.store", "memory.recall"):
                await self._store_action_episode(request, result, context)

            return result

        except Exception as e:
            logger.error(f"Action handler error: {e}")
            request.status = ActionStatus.FAILED
            return VoiceActionResult(
                success=False,
                spoken_response=f"Sorry, there was an error: {e!s}",
            )

    async def _request_confirmation(
        self,
        request: ActionRequest,
        action_def: ActionDefinition | None,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Request voice confirmation for risky action."""
        # Build confirmation prompt
        if action_def and action_def.confirmation_template:
            try:
                prompt = action_def.confirmation_template.format(**request.parameters)
            except KeyError:
                prompt = f"Do you want me to {request.action_name.replace('.', ' ')}?"
        else:
            prompt = f"Should I {request.action_name.replace('.', ' ')}?"

        # Store pending confirmation
        self._pending_confirmations[context.session_id] = request
        request.status = ActionStatus.AWAITING_CONFIRMATION

        return VoiceActionResult(
            success=False,
            spoken_response=prompt,
            requires_confirmation=True,
            confirmation_prompt=prompt,
        )

    async def _handle_confirmation_response(
        self,
        response_request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Handle response to confirmation request."""
        pending = self._pending_confirmations.pop(context.session_id, None)
        if not pending:
            return VoiceActionResult(
                success=False,
                spoken_response="I'm not sure what you're confirming.",
            )

        # Check if confirmed or cancelled
        text = response_request.parameters.get("message", "").lower()

        if any(phrase in text for phrase in self.CONFIRM_PHRASES):
            pending.confirmed = True
            return await self.route(pending, context)

        if any(phrase in text for phrase in self.CANCEL_PHRASES):
            return VoiceActionResult(
                success=True,
                spoken_response="Cancelled.",
            )

        # Unclear response - ask again
        self._pending_confirmations[context.session_id] = pending
        return VoiceActionResult(
            success=False,
            spoken_response="I didn't catch that. Please say yes or no.",
            requires_confirmation=True,
        )

    async def _store_action_episode(
        self,
        request: ActionRequest,
        result: VoiceActionResult,
        context: VoiceContext,
    ) -> None:
        """Store successful action as episode."""
        content = f"Action: {request.action_name} | "
        content += f"Parameters: {request.parameters} | "
        content += f"Result: {result.spoken_response[:100]}"

        await self.memory.on_user_speech(content, context, store_immediately=True)

    # =========================================================================
    # Email Handlers
    # =========================================================================

    async def _handle_email_send(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Send an email."""
        recipient = request.parameters.get("recipient", "")
        subject = request.parameters.get("subject", "")
        body = request.parameters.get("body", "")

        if not recipient:
            return VoiceActionResult(
                success=False,
                spoken_response="Who should I send the email to?",
            )

        # Resolve contact name to email
        email = await self._resolve_contact_email(recipient)
        if not email:
            return VoiceActionResult(
                success=False,
                spoken_response=f"I couldn't find an email address for {recipient}.",
            )

        # Build email body if not provided
        if not body and subject:
            body = f"Subject: {subject}\n\n[Voice message - body to be added]"

        result = await self.mcp.call_tool(
            "mcp__google-workspace__gmail_send_email",
            {
                "to": email,
                "subject": subject or "Voice message",
                "body": body,
            }
        )

        return VoiceActionResult(
            success=True,
            spoken_response=f"Email sent to {recipient}.",
            data=result,
        )

    async def _handle_email_read(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Read recent/unread emails."""
        result = await self.mcp.call_tool(
            "mcp__google-workspace__gmail_list_emails",
            {
                "hours": 24,
                "maxResults": 5,
                "query": "is:unread",
            }
        )

        messages = result.get("messages", [])
        if not messages:
            return VoiceActionResult(
                success=True,
                spoken_response="You have no unread emails.",
            )

        # Summarize for voice
        summaries = []
        for msg in messages[:3]:
            sender = msg.get("from", "Unknown sender")
            # Extract just the name part
            if "<" in sender:
                sender = sender.split("<")[0].strip()
            subject = msg.get("subject", "No subject")
            summaries.append(f"From {sender}: {subject}")

        count = len(messages)
        response = f"You have {count} unread emails. " + ". ".join(summaries)

        return VoiceActionResult(
            success=True,
            spoken_response=response,
            data=messages,
        )

    async def _handle_email_search(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Search emails."""
        from_person = request.parameters.get("from", "")
        query = f"from:{from_person}" if from_person else ""

        result = await self.mcp.call_tool(
            "mcp__google-workspace__gmail_search_emails",
            {
                "query": query,
                "maxResults": 5,
            }
        )

        messages = result.get("messages", [])
        if not messages:
            return VoiceActionResult(
                success=True,
                spoken_response=f"No emails found from {from_person}.",
            )

        return VoiceActionResult(
            success=True,
            spoken_response=f"Found {len(messages)} emails from {from_person}.",
            data=messages,
        )

    async def _handle_email_reply(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Reply to an email."""
        # This would need more context about which email to reply to
        return VoiceActionResult(
            success=False,
            spoken_response="I need to know which email to reply to. Can you be more specific?",
        )

    # =========================================================================
    # Calendar Handlers
    # =========================================================================

    async def _handle_calendar_create(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Create a calendar event."""
        from t4dm.integrations.kymera.intent_parser import TimeParser

        title = request.parameters.get("title", "Meeting")
        when = request.parameters.get("when", "")

        # Parse time
        parser = TimeParser()
        start_time = parser.parse(when)

        if not start_time:
            return VoiceActionResult(
                success=False,
                spoken_response="When should I schedule this?",
            )

        # Default 1 hour duration
        from datetime import timedelta
        end_time = start_time + timedelta(hours=1)

        result = await self.mcp.call_tool(
            "mcp__google-workspace__calendar_create_event",
            {
                "summary": title,
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            }
        )

        time_str = start_time.strftime("%A at %I:%M %p")
        return VoiceActionResult(
            success=True,
            spoken_response=f"I've scheduled {title} for {time_str}.",
            data=result,
        )

    async def _handle_calendar_list(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """List calendar events."""
        from datetime import datetime

        result = await self.mcp.call_tool(
            "mcp__google-workspace__calendar_list_events",
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "days": 1,
                "maxResults": 5,
            }
        )

        events = result.get("items", [])
        if not events:
            return VoiceActionResult(
                success=True,
                spoken_response="Your calendar is clear today.",
            )

        # Format for voice
        summaries = []
        for event in events[:5]:
            title = event.get("summary", "Untitled event")
            start = event.get("start", {}).get("dateTime", "")
            if start:
                try:
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    time_str = dt.strftime("%I:%M %p")
                    summaries.append(f"At {time_str}, {title}")
                except ValueError:
                    summaries.append(title)
            else:
                summaries.append(title)

        response = f"You have {len(events)} events today. " + ". ".join(summaries)

        return VoiceActionResult(
            success=True,
            spoken_response=response,
            data=events,
        )

    async def _handle_calendar_query(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Query calendar for specific event."""
        query = request.parameters.get("query", "")

        # Search in upcoming events
        from datetime import datetime

        result = await self.mcp.call_tool(
            "mcp__google-workspace__calendar_list_events",
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "days": 30,
                "maxResults": 20,
            }
        )

        events = result.get("items", [])

        # Find matching event
        for event in events:
            title = event.get("summary", "").lower()
            if query.lower() in title:
                start = event.get("start", {}).get("dateTime", "")
                if start:
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    time_str = dt.strftime("%A, %B %d at %I:%M %p")
                    return VoiceActionResult(
                        success=True,
                        spoken_response=f"Your {event.get('summary')} is on {time_str}.",
                        data=event,
                    )

        return VoiceActionResult(
            success=True,
            spoken_response=f"I couldn't find any events matching {query}.",
        )

    # =========================================================================
    # Memory Handlers
    # =========================================================================

    async def _handle_memory_store(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Store something in memory."""
        content = request.parameters.get("content", "")

        if not content:
            return VoiceActionResult(
                success=False,
                spoken_response="What should I remember?",
            )

        episode_id = await self.memory.store_explicit_memory(content, context)

        return VoiceActionResult(
            success=True,
            spoken_response="I'll remember that.",
            data={"episode_id": str(episode_id)},
        )

    async def _handle_memory_recall(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Recall from memory."""
        query = request.parameters.get("query", "")

        if not query:
            return VoiceActionResult(
                success=False,
                spoken_response="What do you want me to recall?",
            )

        memories = await self.memory.recall_explicit(query, context)

        if not memories:
            return VoiceActionResult(
                success=True,
                spoken_response=f"I don't have any memories about {query}.",
            )

        # Summarize memories
        summaries = []
        for mem in memories[:3]:
            content = mem.get("content", "")[:100]
            summaries.append(content)

        response = f"Here's what I remember about {query}: " + ". ".join(summaries)

        return VoiceActionResult(
            success=True,
            spoken_response=response,
            data=memories,
        )

    # =========================================================================
    # Other Handlers
    # =========================================================================

    async def _handle_reminder_set(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Set a reminder."""
        message = request.parameters.get("message", "")
        when = request.parameters.get("when", "")

        # For now, store as memory episode with context
        content = f"Reminder: {message}"
        if when:
            content += f" (at {when})"

        await self.memory.on_user_speech(content, context, store_immediately=True)

        return VoiceActionResult(
            success=True,
            spoken_response=f"I'll remind you to {message}.",
        )

    async def _handle_reminder_list(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """List reminders."""
        memories = await self.memory.recall_explicit("reminder", context)

        if not memories:
            return VoiceActionResult(
                success=True,
                spoken_response="You don't have any active reminders.",
            )

        return VoiceActionResult(
            success=True,
            spoken_response=f"You have {len(memories)} reminders.",
            data=memories,
        )

    async def _handle_task_create(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Create a task."""
        title = request.parameters.get("title", "")

        content = f"Task: {title}"
        await self.memory.on_user_speech(content, context, store_immediately=True)

        return VoiceActionResult(
            success=True,
            spoken_response=f"Added task: {title}.",
        )

    async def _handle_task_list(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """List tasks."""
        memories = await self.memory.recall_explicit("task", context)

        if not memories:
            return VoiceActionResult(
                success=True,
                spoken_response="You don't have any tasks.",
            )

        return VoiceActionResult(
            success=True,
            spoken_response=f"You have {len(memories)} tasks.",
            data=memories,
        )

    async def _handle_task_complete(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Complete a task."""
        title = request.parameters.get("title", "")

        return VoiceActionResult(
            success=True,
            spoken_response=f"Marked {title} as complete.",
        )

    async def _handle_contact_lookup(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Look up contact information."""
        name = request.parameters.get("name", "")

        # Search in memory
        memories = await self.memory.recall_explicit(name, context)

        if not memories:
            return VoiceActionResult(
                success=True,
                spoken_response=f"I don't have any information about {name}.",
            )

        # Return what we know
        info = memories[0].get("content", "")[:200]
        return VoiceActionResult(
            success=True,
            spoken_response=f"About {name}: {info}",
            data=memories,
        )

    async def _handle_lookup_time(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Get current time."""
        from datetime import datetime
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")

        return VoiceActionResult(
            success=True,
            spoken_response=f"It's {time_str}.",
        )

    async def _handle_lookup_date(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Get current date."""
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%A, %B %d")

        return VoiceActionResult(
            success=True,
            spoken_response=f"It's {date_str}.",
        )

    async def _handle_claude_chat(
        self,
        request: ActionRequest,
        context: VoiceContext,
    ) -> VoiceActionResult:
        """Fall through to Claude for general chat."""
        message = request.parameters.get("message", "")

        # Get memory context
        await self.memory.get_relevant_context(message, context)

        # Build enhanced prompt
        from t4dm.integrations.kymera.context_injector import ContextInjector
        injector = ContextInjector(self.memory)
        system_prompt = await injector.build_system_prompt(message, context)

        # Call Claude
        response = await self.claude.chat(message, system_prompt=system_prompt)

        return VoiceActionResult(
            success=True,
            spoken_response=response,
        )

    async def _resolve_contact_email(self, name: str) -> str | None:
        """Resolve contact name to email address."""
        # Try memory first
        memories = await self.memory.recall_explicit(f"{name} email", VoiceContext(session_id="system"))

        for mem in memories:
            content = mem.get("content", "")
            # Simple email extraction
            import re
            match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", content)
            if match:
                return match.group(0)

        return None
