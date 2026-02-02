# Kymera Voice + T4DM Integration Plan

**Full AI Assistant Architecture**

## Executive Summary

Integrate T4DM's tripartite memory system with Kymera Voice to create a complete AI assistant with:
- Persistent memory across conversations
- Smart action execution with confirmations
- Personal data (calendar, email, contacts) integration
- Proactive context surfacing
- Privacy-aware voice interactions

---

## Current State Analysis

### Kymera Voice (~/vm/kymera-voice)
| Component | Status | Technology |
|-----------|--------|------------|
| Wake Word | Complete | OpenWakeWord |
| STT | Complete | faster-whisper |
| TTS | Complete | Kokoro |
| VAD | Complete | Silero |
| Claude API | Complete | Direct API + CLI |
| System Actions | Basic | pactl, playerctl, xdotool |
| MCP Server | Complete | voice_speak, voice_listen |
| Conversation Manager | Complete | State machine |

### T4DM (~/ww)
| Component | Status | Technology |
|-----------|--------|------------|
| Episodic Memory | Complete | Qdrant vectors |
| Semantic Memory | Complete | Neo4j graph |
| Procedural Memory | Complete | Skills/Procedures |
| Memory Gate | NEW | Score-based filtering |
| Privacy Filter | NEW | PII redaction |
| Personal Entities | NEW | Contact, Event, Email, Task |
| Action Framework | NEW | 60+ defined actions |
| Google Workspace | NEW | Calendar, Gmail, Drive sync |

### Integration Gap
```
┌─────────────────────────────────────────────────────────────┐
│                    MISSING LAYERS                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Intent Parser - Voice → Structured Action Request        │
│ 2. Memory Bridge - Voice ↔ WW Memory                        │
│ 3. Context Provider - Proactive memory surfacing            │
│ 4. Action Router - Route to correct handler                 │
│ 5. Confirmation Flow - Voice confirmation for risky actions │
│ 6. Continuous Capture - Store conversation episodes         │
│ 7. Personal Data Query - "What's on my calendar?"           │
└─────────────────────────────────────────────────────────────┘
```

---

## Target Architecture

```
                         ┌──────────────────────────┐
                         │      Voice Input         │
                         │   (Wake Word + STT)      │
                         └───────────┬──────────────┘
                                     │
                         ┌───────────▼──────────────┐
                         │     Privacy Filter       │
                         │   (Redact PII, "off      │
                         │    the record")          │
                         └───────────┬──────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
    │   Memory Gate     │  │  Intent Parser  │  │ Context Provider  │
    │   (Store/Skip)    │  │  (NLU → Action) │  │ (Recall relevant) │
    └─────────┬─────────┘  └────────┬────────┘  └─────────┬─────────┘
              │                     │                     │
              │            ┌────────▼────────┐            │
              │            │  Action Router  │◄───────────┘
              │            │  (Permission    │
              │            │   + Confirm)    │
              │            └────────┬────────┘
              │                     │
    ┌─────────▼─────────┐  ┌────────▼────────────────────────────────┐
    │  T4DM     │  │            Action Handlers              │
    │  ┌─────────────┐  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐│
    │  │ Episodes    │  │  │  │ Email    │ │ Calendar │ │ System   ││
    │  │ Entities    │  │  │  │ Handler  │ │ Handler  │ │ Handler  ││
    │  │ Skills      │  │  │  └──────────┘ └──────────┘ └──────────┘│
    │  └─────────────┘  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐│
    └───────────────────┘  │  │ Claude   │ │ Smart    │ │ Memory   ││
                           │  │ Code CLI │ │ Home     │ │ Ops      ││
                           │  └──────────┘ └──────────┘ └──────────┘│
                           └────────────────────────────────────────┘
                                     │
                         ┌───────────▼──────────────┐
                         │   Response Synthesizer   │
                         │   (Summarize for voice)  │
                         └───────────┬──────────────┘
                                     │
                         ┌───────────▼──────────────┐
                         │      Voice Output        │
                         │    (Streaming TTS)       │
                         └──────────────────────────┘
```

---

## Phase 1: Core Integration Bridge

**Duration**: Foundation layer
**Agent**: `ww-conductor`, `kymera-electron-dev`

### 1.1 Create Integration Package

Create `~/t4dm/src/t4dm/integrations/kymera/` with:

```
kymera/
├── __init__.py
├── bridge.py           # Main integration class
├── memory_hooks.py     # Hooks for voice → memory
├── context_injector.py # Inject memory into prompts
└── voice_actions.py    # Voice-aware action execution
```

### 1.2 Memory Bridge

```python
class VoiceMemoryBridge:
    """Connect Kymera Voice to T4DM memory."""

    def __init__(self, ww_client, memory_gate, privacy_filter):
        self.ww = ww_client
        self.gate = memory_gate
        self.privacy = privacy_filter

    async def on_user_speech(self, text: str, context: VoiceContext):
        """Called when user speaks - decide if should store."""
        # Filter first
        result = self.privacy.filter(text)
        if result.blocked:
            return

        # Gate decision
        gate_ctx = GateContext(
            session_id=context.session_id,
            project=context.project,
            is_voice=True,
        )
        decision = self.gate.evaluate(result.content, gate_ctx)

        if decision.decision == StorageDecision.STORE:
            await self.ww.store_episode(
                content=result.content,
                context=context.to_dict(),
                importance=decision.suggested_importance,
            )

    async def on_assistant_response(self, text: str, spoken: str):
        """Store assistant response if significant."""
        # Usually don't store assistant responses
        # Unless they contain learned information
        pass

    async def get_relevant_context(self, query: str) -> str:
        """Get memory context for current query."""
        episodes = await self.ww.recall_episodes(query, limit=5)
        entities = await self.ww.semantic_recall(query, limit=10)
        skills = await self.ww.recall_skills(query, limit=3)

        return self._format_context(episodes, entities, skills)
```

### 1.3 Context Injection

Modify Jarvis to inject memory context into Claude prompts:

```python
# In jarvis.py _respond method
async def _respond(self, user_text: str) -> None:
    # Get memory context
    memory_context = await self.memory_bridge.get_relevant_context(user_text)

    # Inject into system prompt
    enhanced_prompt = f"""
{self.SYSTEM_PROMPT}

## Memory Context
{memory_context}
"""
    # Use enhanced prompt for Claude call
```

### 1.4 Deliverables
- [ ] `VoiceMemoryBridge` class
- [ ] `ContextInjector` for prompts
- [ ] Hook integration in Jarvis
- [ ] Unit tests

---

## Phase 2: Intent Parser & Action Router

**Duration**: NLU layer
**Agent**: `ww-planner`, `ww-algorithm`

### 2.1 Intent Classification

Create intent parser that maps voice input to structured actions:

```python
class VoiceIntentParser:
    """Parse voice input into actionable intents."""

    INTENT_PATTERNS = {
        # Email
        r"(send|write|compose)\s+(an?\s+)?email\s+to\s+(.+)": "email.send",
        r"read\s+(my\s+)?(latest|recent|new)\s+emails?": "email.read",
        r"reply\s+to\s+(.+)": "email.reply",

        # Calendar
        r"(schedule|create|add)\s+(a\s+)?meeting\s+(.+)": "calendar.create",
        r"what('s| is)\s+(on\s+)?(my\s+)?calendar": "calendar.list",
        r"when\s+is\s+(my\s+)?next\s+(.+)": "calendar.query",

        # Tasks
        r"remind\s+me\s+to\s+(.+)": "reminder.set",
        r"add\s+(to\s+)?(my\s+)?to(-)?do": "task.create",
        r"what('s| are)\s+(on\s+)?(my\s+)?tasks?": "task.list",

        # Memory
        r"remember\s+(that\s+)?(.+)": "memory.store",
        r"what\s+do\s+you\s+(know|remember)\s+about\s+(.+)": "memory.recall",
        r"forget\s+(.+)": "memory.forget",

        # Personal
        r"who\s+is\s+(.+)": "contact.lookup",
        r"(what's|when is)\s+(.+)'s\s+birthday": "contact.birthday",
    }

    async def parse(self, text: str, context: VoiceContext) -> ActionRequest:
        """Parse voice input to action request."""
        # Try pattern matching first (fast)
        for pattern, action_name in self.INTENT_PATTERNS.items():
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return self._build_request(action_name, match, context)

        # Fall back to Claude for complex intents
        return await self._llm_parse(text, context)

    async def _llm_parse(self, text: str, context: VoiceContext) -> ActionRequest:
        """Use Claude to parse complex intents."""
        prompt = f"""
Parse this voice command into a structured action.

Command: "{text}"

Available actions: {list(BUILTIN_ACTIONS.keys())}

Return JSON:
{{
  "action": "action.name",
  "parameters": {{...}},
  "confidence": 0.0-1.0
}}

If no action matches, return {{"action": "claude.chat", "parameters": {{"message": "{text}"}}}}.
"""
        # Call Claude with JSON mode
        result = await self.claude.parse(prompt)
        return ActionRequest(**result)
```

### 2.2 Action Router

Route parsed intents to appropriate handlers:

```python
class VoiceActionRouter:
    """Route voice intents to action handlers."""

    def __init__(self, registry: ActionRegistry, executor: ActionExecutor):
        self.registry = registry
        self.executor = executor
        self.handlers: dict[str, Callable] = {}

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        # Email handlers (Google Workspace MCP)
        self.handlers["email.send"] = self._handle_email_send
        self.handlers["email.read"] = self._handle_email_read

        # Calendar handlers
        self.handlers["calendar.create"] = self._handle_calendar_create
        self.handlers["calendar.list"] = self._handle_calendar_list

        # Memory handlers (T4DM)
        self.handlers["memory.store"] = self._handle_memory_store
        self.handlers["memory.recall"] = self._handle_memory_recall

        # Default: Claude chat
        self.handlers["claude.chat"] = self._handle_claude_chat

    async def route(self, request: ActionRequest) -> ActionResult:
        """Route action to handler."""
        handler = self.handlers.get(request.action_name)

        if not handler:
            # Fall through to Claude
            handler = self.handlers["claude.chat"]

        # Check permissions
        permission = self.registry.get_permission(request.action_name)
        if permission == PermissionLevel.BLOCKED:
            return ActionResult(
                success=False,
                message="This action is not permitted",
            )

        # Handle confirmation if needed
        if permission in (PermissionLevel.CONFIRM, PermissionLevel.VERIFY):
            if not request.confirmed:
                return await self._request_confirmation(request)

        return await handler(request)
```

### 2.3 Voice Confirmation Flow

```python
class VoiceConfirmation:
    """Handle voice confirmation for risky actions."""

    CONFIRM_PHRASES = ["yes", "confirm", "do it", "go ahead", "send it"]
    CANCEL_PHRASES = ["no", "cancel", "stop", "nevermind", "don't"]

    async def request_confirmation(
        self,
        request: ActionRequest,
        voice_output: StreamingVoiceOutput,
        voice_input: WhisperSTT,
    ) -> bool:
        """Request voice confirmation."""
        # Get confirmation prompt
        action = self.registry.get(request.action_name)
        prompt = action.confirmation_template.format(**request.parameters)

        # Speak confirmation prompt
        voice_output.speak(prompt)

        # Listen for response
        audio = await self._listen(timeout=10.0)
        text = voice_input.transcribe(audio).text.lower()

        if any(phrase in text for phrase in self.CONFIRM_PHRASES):
            return True
        if any(phrase in text for phrase in self.CANCEL_PHRASES):
            voice_output.speak("Cancelled.")
            return False

        # Unclear response
        voice_output.speak("I didn't understand. Please say yes or no.")
        return await self.request_confirmation(request, voice_output, voice_input)
```

### 2.4 Deliverables
- [ ] `VoiceIntentParser` with patterns + LLM fallback
- [ ] `VoiceActionRouter` with handler registration
- [ ] `VoiceConfirmation` flow
- [ ] Integration with existing Jarvis system commands

---

## Phase 3: Personal Data Integration

**Duration**: Data sync layer
**Agent**: `ww-memory`, `ww-retriever`

### 3.1 Data Sync Service

```python
class PersonalDataSyncService:
    """Background service for syncing personal data."""

    def __init__(self, google_sync: GoogleWorkspaceSync, ww_store: PersonalDataStore):
        self.google = google_sync
        self.store = ww_store
        self._running = False

    async def start(self):
        """Start background sync."""
        self._running = True

        # Initial sync
        await self.sync_all()

        # Periodic sync
        while self._running:
            await asyncio.sleep(300)  # Every 5 minutes
            await self.sync_incremental()

    async def sync_all(self):
        """Full sync of all data."""
        # Sync calendar
        events = await self.google.sync_calendar(days=30)
        events = await self.google.embed_events(events)
        for event in events:
            await self.store.store_event(event)

        # Sync emails
        emails = await self.google.sync_emails(hours=168)  # 7 days
        emails = await self.google.embed_emails(emails)
        for email in emails:
            await self.store.store_email(email)

        # Extract contacts
        contacts = await self.google.extract_contacts_from_emails(emails)
        contacts = await self.google.embed_contacts(contacts)
        for contact in contacts:
            await self.store.store_contact(contact)
```

### 3.2 Voice Query Handlers

```python
class CalendarVoiceHandler:
    """Handle calendar voice queries."""

    async def handle_list(self, request: ActionRequest) -> ActionResult:
        """What's on my calendar?"""
        events = await self.store.search_events(
            query="",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1),
        )

        if not events:
            return ActionResult(
                success=True,
                spoken_response="Your calendar is clear today.",
            )

        # Format for speech
        summaries = []
        for e in events[:5]:
            time_str = e.start.strftime("%I:%M %p")
            summaries.append(f"At {time_str}, {e.title}")

        return ActionResult(
            success=True,
            spoken_response=f"You have {len(events)} events today. " + ". ".join(summaries),
        )

    async def handle_create(self, request: ActionRequest) -> ActionResult:
        """Schedule a meeting..."""
        # Parse time from natural language
        parsed = await self._parse_event_time(request.parameters.get("when"))

        result = await self.mcp(
            "mcp__google-workspace__calendar_create_event",
            {
                "summary": request.parameters.get("title"),
                "start": parsed.start.isoformat(),
                "end": parsed.end.isoformat(),
            }
        )

        return ActionResult(
            success=True,
            spoken_response=f"I've scheduled {request.parameters.get('title')} for {parsed.start.strftime('%A at %I:%M %p')}.",
        )


class EmailVoiceHandler:
    """Handle email voice queries."""

    async def handle_read(self, request: ActionRequest) -> ActionResult:
        """Read my latest emails."""
        emails = await self.store.search_emails(
            query="is:unread",
            limit=5,
        )

        if not emails:
            return ActionResult(
                success=True,
                spoken_response="You have no unread emails.",
            )

        # Summarize for voice
        summaries = []
        for e in emails[:3]:
            sender = e.from_address.name or e.from_address.email.split("@")[0]
            summaries.append(f"From {sender}, subject: {e.subject}")

        return ActionResult(
            success=True,
            spoken_response=f"You have {len(emails)} unread emails. " + ". ".join(summaries),
        )


class ContactVoiceHandler:
    """Handle contact voice queries."""

    async def handle_lookup(self, request: ActionRequest) -> ActionResult:
        """Who is John Smith?"""
        name = request.parameters.get("name")
        contacts = await self.store.search_contacts(name, limit=1)

        if not contacts:
            return ActionResult(
                success=True,
                spoken_response=f"I don't have anyone named {name} in your contacts.",
            )

        c = contacts[0]
        parts = [f"{c.name}"]
        if c.organization:
            parts.append(f"works at {c.organization}")
        if c.job_title:
            parts.append(f"as {c.job_title}")
        if c.relationship.value != "other":
            parts.append(f"is your {c.relationship.value}")

        return ActionResult(
            success=True,
            spoken_response=" ".join(parts) + ".",
        )
```

### 3.3 Deliverables
- [ ] `PersonalDataSyncService` with background sync
- [ ] `CalendarVoiceHandler`
- [ ] `EmailVoiceHandler`
- [ ] `ContactVoiceHandler`
- [ ] Integration with voice router

---

## Phase 4: Memory Continuity

**Duration**: Conversation memory
**Agent**: `ww-synthesizer`, `ww-session`

### 4.1 Conversation Episode Capture

```python
class ConversationCapture:
    """Capture conversation as episodes."""

    def __init__(self, memory_bridge: VoiceMemoryBridge, batcher: TemporalBatcher):
        self.bridge = memory_bridge
        self.batcher = batcher
        self._conversation_buffer: list[Message] = []

    async def on_turn_complete(self, user_msg: str, assistant_msg: str):
        """Called after each conversation turn."""
        self._conversation_buffer.append(Message("user", user_msg))
        self._conversation_buffer.append(Message("assistant", assistant_msg))

        # Batch and potentially store
        batch_key = f"conversation:{datetime.now().strftime('%Y%m%d_%H')}"
        batched = self.batcher.add(batch_key, f"User: {user_msg}\nAssistant: {assistant_msg}")

        if batched:
            await self._store_conversation_batch(batched)

    async def on_conversation_end(self):
        """Called when conversation ends - store summary."""
        if not self._conversation_buffer:
            return

        # Generate summary
        summary = await self._summarize_conversation()

        # Store as episode
        await self.bridge.ww.store_episode(
            content=summary,
            context={
                "type": "voice_conversation",
                "turns": len(self._conversation_buffer) // 2,
                "duration": self._conversation_duration,
            },
            importance=0.6,
        )

        self._conversation_buffer.clear()

    async def _summarize_conversation(self) -> str:
        """Summarize conversation for storage."""
        # Use Claude to summarize
        messages = "\n".join(
            f"{m.role}: {m.content}"
            for m in self._conversation_buffer
        )

        prompt = f"""
Summarize this voice conversation in 2-3 sentences.
Focus on: what was discussed, decisions made, actions taken.

{messages}
"""
        return await self.claude.complete(prompt)
```

### 4.2 Context Surfacing

```python
class ProactiveContextProvider:
    """Surface relevant memories proactively."""

    async def get_context_for_query(self, query: str) -> MemoryContext:
        """Get context based on query."""
        # Semantic search for related episodes
        episodes = await self.ww.recall_episodes(query, limit=5)

        # Spread activation for related entities
        entities = await self.ww.semantic_recall(query, limit=10)
        activated = await self.ww.spread_activation(
            seed_ids=[e.id for e in entities[:3]],
            max_depth=2,
        )

        # Check for applicable skills
        skills = await self.ww.recall_skills(query, limit=3)

        return MemoryContext(
            episodes=episodes,
            entities=activated,
            skills=skills,
        )

    async def get_proactive_context(self) -> Optional[str]:
        """Get proactive context based on time/calendar."""
        now = datetime.now()

        # Check upcoming events
        events = await self.store.search_events(
            query="",
            start_date=now,
            end_date=now + timedelta(hours=2),
        )

        if events:
            next_event = events[0]
            minutes_until = (next_event.start - now).total_seconds() / 60

            if minutes_until <= 15:
                # Get context about attendees
                attendee_context = await self._get_attendee_context(next_event)
                return f"Reminder: {next_event.title} starts in {int(minutes_until)} minutes. {attendee_context}"

        # Check overdue tasks
        tasks = await self.store.search_tasks(status=TaskStatus.TODO, overdue=True)
        if tasks:
            return f"You have {len(tasks)} overdue tasks."

        return None
```

### 4.3 Cross-Session Memory

```python
class SessionMemory:
    """Manage memory across voice sessions."""

    async def load_session_context(self, session_id: str) -> str:
        """Load context for new session."""
        # Get last session summary
        last_episodes = await self.ww.recall_episodes(
            query="voice_conversation",
            limit=3,
            time_filter={"after": "24 hours ago"},
        )

        # Get any pending actions
        pending = await self.ww.recall_episodes(
            query="pending action reminder",
            limit=5,
        )

        # Get upcoming events
        events = await self.store.search_events(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(hours=24),
        )

        context_parts = []

        if last_episodes:
            context_parts.append("Recent conversations: " +
                "; ".join(e.content[:100] for e in last_episodes))

        if pending:
            context_parts.append("Pending: " +
                "; ".join(e.content for e in pending))

        if events:
            context_parts.append(f"Upcoming: {len(events)} events today")

        return "\n".join(context_parts)

    async def save_session_summary(self, session_id: str, summary: str):
        """Save session summary for future context."""
        await self.ww.store_episode(
            content=summary,
            context={"type": "session_summary", "session_id": session_id},
            importance=0.5,
        )
```

### 4.4 Deliverables
- [ ] `ConversationCapture` with batching
- [ ] `ProactiveContextProvider`
- [ ] `SessionMemory` for cross-session
- [ ] Integration with Jarvis lifecycle

---

## Phase 5: Advanced Features

**Duration**: Polish and extensions
**Agent**: `ww-conductor`, `ww-validator`

### 5.1 Proactive Notifications

```python
class VoiceNotificationService:
    """Proactive voice notifications."""

    async def check_and_notify(self):
        """Check for things to notify about."""
        # Calendar reminders
        upcoming = await self._check_upcoming_events()
        if upcoming:
            await self.notify(upcoming)

        # Birthday reminders
        birthdays = await self._check_birthdays()
        if birthdays:
            await self.notify(birthdays)

        # Important emails
        urgent_emails = await self._check_urgent_emails()
        if urgent_emails:
            await self.notify(urgent_emails)

    async def notify(self, message: str):
        """Speak notification."""
        # Check if in conversation (don't interrupt)
        if self.jarvis.conversation.is_active:
            # Queue for after conversation
            self._notification_queue.append(message)
            return

        # Use attention sound + speak
        await self.voice_output.play_attention_sound()
        await self.voice_output.speak(message)
```

### 5.2 Learning from Feedback

```python
class FeedbackLearner:
    """Learn from user corrections and feedback."""

    CORRECTION_PATTERNS = [
        r"no,?\s+i\s+(meant|said|want)",
        r"that's\s+not\s+(right|correct|what i)",
        r"actually,?\s+i\s+",
    ]

    async def on_user_speech(self, text: str, previous_response: str):
        """Check for correction patterns."""
        if self._is_correction(text):
            # Store as learning episode
            await self.ww.store_episode(
                content=f"Correction: User said '{text}' after I said '{previous_response}'",
                context={"type": "correction"},
                importance=0.8,
            )

            # Update entity if applicable
            # ...

    async def on_action_feedback(self, action: str, success: bool, feedback: str):
        """Learn from action outcomes."""
        if success:
            # Strengthen skill if applicable
            skill = await self.ww.find_skill(action)
            if skill:
                skill.success_rate = (skill.success_rate * skill.execution_count + 1) / (skill.execution_count + 1)
                skill.execution_count += 1
                await self.ww.update_skill(skill)
        else:
            # Store failure for learning
            await self.ww.store_episode(
                content=f"Action '{action}' failed: {feedback}",
                context={"type": "action_failure"},
                importance=0.7,
                outcome=Outcome.FAILURE,
            )
```

### 5.3 Multi-Modal Context

```python
class ScreenContextProvider:
    """Get context from current screen (for desktop assistant)."""

    async def get_screen_context(self) -> str:
        """Get context about what user is looking at."""
        # Get active window
        window_title = await self._get_active_window_title()

        # Get context based on app
        if "code" in window_title.lower() or "vim" in window_title.lower():
            # They're coding
            return "User is currently coding"
        elif "browser" in window_title.lower():
            # Get page title
            return f"User is browsing: {window_title}"
        elif "slack" in window_title.lower() or "discord" in window_title.lower():
            return "User is in a chat application"

        return f"Active window: {window_title}"
```

### 5.4 Deliverables
- [ ] `VoiceNotificationService`
- [ ] `FeedbackLearner`
- [ ] `ScreenContextProvider`
- [ ] Performance optimizations
- [ ] Documentation

---

## Implementation Agents

| Phase | Primary Agent | Support Agents |
|-------|--------------|----------------|
| 1 | `ww-conductor` | `kymera-go-backend` |
| 2 | `ww-planner` | `ww-algorithm` |
| 3 | `ww-memory` | `ww-retriever` |
| 4 | `ww-synthesizer` | `ww-session` |
| 5 | `ww-conductor` | `ww-validator` |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Voice-to-action latency | < 500ms |
| Memory recall accuracy | > 85% |
| Action success rate | > 95% |
| Context relevance | > 80% |
| User satisfaction | > 4/5 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Privacy leaks | PrivacyFilter mandatory, audit logging |
| Action mistakes | Confirmation required for HIGH/CRITICAL |
| Memory bloat | MemoryGate filtering, consolidation |
| Latency | Caching, async operations, local models |
| Sync failures | Retry logic, graceful degradation |

---

## Timeline Considerations

Each phase should be implemented sequentially, with Phase 1 being the foundation. Phases can be parallelized once Phase 1 is complete:
- Phase 2 + Phase 3 can run in parallel
- Phase 4 depends on Phase 2 + 3
- Phase 5 can start after Phase 4 basics

---

## Next Steps

1. Create integration package structure
2. Implement `VoiceMemoryBridge`
3. Test with existing Jarvis pipeline
4. Iterate based on feedback
