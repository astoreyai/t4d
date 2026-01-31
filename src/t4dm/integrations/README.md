# Integrations Module

**11 files | ~6,000 lines | Centrality: 2**

The integrations module provides adapters connecting World Weaver to external services including Kymera Voice and Google Workspace.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       INTEGRATIONS LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    KYMERA VOICE INTEGRATION                         ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  ││
│  │  │ Intent Parser│  │Action Router │  │   Memory Bridge          │  ││
│  │  │ Regex + LLM  │  │ 21 handlers  │  │   Privacy + Batching     │  ││
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  ││
│  │         │                 │                     │                   ││
│  │         └─────────────────┼─────────────────────┘                   ││
│  │                           ▼                                         ││
│  │              ┌────────────────────────┐                             ││
│  │              │  Voice Action Executor │                             ││
│  │              │  Jarvis Memory Hook    │                             ││
│  │              └────────────────────────┘                             ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                  GOOGLE WORKSPACE INTEGRATION                       ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  ││
│  │  │Gmail Sync    │  │Calendar Sync │  │   Personal Data Store    │  ││
│  │  │Email → Entity│  │Events → Embed│  │   Neo4j + Qdrant         │  ││
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `google_workspace.py` | ~685 | Gmail, Calendar, Drive sync |
| `kymera/bridge.py` | ~535 | Voice-to-memory bridge |
| `kymera/action_router.py` | ~775 | Voice intent routing |
| `kymera/intent_parser.py` | ~410 | Pattern + LLM parsing |
| `kymera/jarvis_hook.py` | ~490 | Jarvis integration point |
| `kymera/voice_actions.py` | ~460 | Main executor |
| `kymera/context_injector.py` | ~375 | Claude prompt enhancement |
| `kymera/memory_continuity.py` | ~735 | Conversation capture |
| `kymera/personal_data.py` | ~520 | Personal data manager |
| `kymera/advanced_features.py` | ~855 | Phase 5 features |
| `kymera/__init__.py` | ~130 | Package exports |

## Kymera Voice Integration

### VoiceMemoryBridge

Central adapter connecting Kymera Voice to World Weaver memory:

```python
from ww.integrations.kymera import VoiceMemoryBridge, VoiceContext

bridge = VoiceMemoryBridge(
    session_id="voice-session",
    store_threshold=0.4,    # MemoryGate threshold
    buffer_threshold=0.2,   # Buffer for batching
    batch_window_seconds=120,
    max_batch_size=10
)

# Process user speech
await bridge.on_user_speech(
    transcription="Remember that the meeting is at 3pm",
    context=VoiceContext(session_id="...", project="work")
)

# Get relevant context for response
memory_context = await bridge.get_relevant_context(
    query="meeting time",
    limit_episodes=5,
    limit_entities=10
)

# Store explicit memory request
await bridge.store_explicit_memory(
    content="Meeting at 3pm tomorrow",
    importance=0.9
)

# End conversation (flush batches)
await bridge.on_conversation_end()
```

**Decision Flow**:
1. Privacy filter content
2. MemoryGate evaluates (STORE/BUFFER/SKIP)
3. STORE → immediate episode creation
4. BUFFER → add to temporal batcher
5. SKIP → discard

### VoiceIntentParser

Two-stage intent parsing:

```python
from ww.integrations.kymera import VoiceIntentParser

parser = VoiceIntentParser()

# Parse voice command
intent = await parser.parse("Send an email to John about the project")
# ParsedIntent(
#     action_name="email.send",
#     parameters={"to": "John", "subject": "the project"},
#     confidence=0.9,
#     entities=["John"]
# )
```

**Pattern Categories** (~100 regex patterns):
- Email: send, read, search, reply, forward
- Calendar: create event, list, query, delete
- Reminders: set, list, cancel
- Tasks: create, list, complete
- Memory: store, recall, forget
- Lookup: time, date, contact

### VoiceActionRouter

Route intents to handlers with permission checking:

```python
from ww.integrations.kymera import VoiceActionRouter, ActionRequest

router = VoiceActionRouter(memory_bridge=bridge)

# Route and execute
result = await router.route(
    request=ActionRequest(
        action_name="email.send",
        parameters={"to": "john@example.com", "subject": "Meeting"}
    ),
    context=voice_context
)

# Permission levels: BLOCKED, CONFIRM, VERIFY, ALLOW
```

**21 Built-in Handlers**:
- Email: send, read, search, reply
- Calendar: create, list, query
- Reminders: set, list
- Tasks: create, list, complete
- Memory: store, recall
- Contacts: lookup
- Lookups: time, date
- Default: Claude chat fallback

### JarvisMemoryHook

Integration point for Kymera Voice's Jarvis:

```python
from ww.integrations.kymera import JarvisMemoryHook

hook = JarvisMemoryHook(session_id="jarvis-session")

# Enhance context before Claude call
memory_context = await hook.enhance_context(query)

# Process speech
await hook.on_user_speech(transcription)

# Process response
await hook.on_response(response_text, action_taken=True)

# Explicit memory ("remember that...")
await hook.on_explicit_memory(content)

# Recall request ("what do you remember about...")
memories = await hook.on_recall_request(query)

# Conversation lifecycle
greeting = await hook.start_conversation()  # With proactive reminders
await hook.end_conversation(store_summary=True)
```

### ContextInjector

Enhance Claude prompts with memory context:

```python
from ww.integrations.kymera import ContextInjector

injector = ContextInjector(
    max_episodes=5,
    max_entities=10,
    max_skills=3,
    max_context_chars=2000
)

# Inject into system prompt
enhanced_prompt = await injector.inject(
    base_prompt="You are a helpful assistant",
    memory_context=memory_context
)
```

**Token Optimization**: ToonJSON encoding (~50% reduction)

**Voice Rules Injected**:
1. Be CONCISE (1-3 sentences)
2. Summarize actions, don't read code
3. Say "about fifty" not "47.3"
4. Use filenames not full paths

## Google Workspace Integration

### GoogleWorkspaceSync

Bidirectional sync with Google services:

```python
from ww.integrations import GoogleWorkspaceSync

sync = GoogleWorkspaceSync(
    session_id="user-session",
    mcp_client=mcp_client
)

# Sync emails
emails = await sync.sync_emails(
    hours=24,
    max_results=50,
    query="is:unread"
)

# Sync calendar
events = await sync.sync_calendar(
    days=7,
    include_past=False
)

# Extract contacts from email interactions
contacts = await sync.extract_contacts_from_emails(emails)

# Build daily context snapshot
context = await sync.build_personal_context()
```

**Features**:
- Privacy filtering (PII redaction)
- Event type classification (MEETING, DEADLINE, TRAVEL, FOCUS_TIME)
- Email importance detection ("urgent", "asap", etc.)
- Contact interaction tracking
- Embedding generation for semantic search

### PersonalDataManager

Unified personal data operations:

```python
from ww.integrations.kymera import PersonalDataManager, PersonalDataConfig

config = PersonalDataConfig(
    email_sync_hours=24,
    calendar_sync_days=7,
    auto_sync_interval_minutes=15,
    enable_birthday_reminders=True,
    enable_meeting_reminders=True
)

manager = PersonalDataManager(config, mcp_client)

# Semantic search
results = await manager.search_contacts("engineering team")
events = await manager.search_events("project deadline")

# Contact resolution (fuzzy matching)
contact = await manager.resolve_contact("John")

# Proactive context
upcoming = await manager.get_proactive_context()
# Upcoming events, overdue tasks, birthdays
```

## Advanced Features (Phase 5)

### NotificationManager

```python
from ww.integrations.kymera import NotificationManager

notifier = NotificationManager(
    quiet_hours_start=22,
    quiet_hours_end=7
)

# Queue notification
await notifier.queue(
    message="Meeting in 15 minutes",
    priority="HIGH",
    scheduled_time=datetime.now() + timedelta(minutes=15)
)

# Voice announcement
announcement = notifier.format_for_voice(notification)
```

### PreferenceLearner

```python
from ww.integrations.kymera import PreferenceLearner

learner = PreferenceLearner()

# Learn from interaction
await learner.learn_preference(
    preference="always_confirm_sensitive_actions",
    context=interaction_context
)

# Apply preferences
should_confirm = await learner.should_confirm(action_type)
```

### VoiceTriggerManager

```python
from ww.integrations.kymera import VoiceTriggerManager

triggers = VoiceTriggerManager()

# Register custom trigger
triggers.register(
    pattern="start my day",
    callback=morning_routine
)
```

## MCP Tool Integration

**Google Workspace MCP Tools**:
- `mcp__google-workspace__gmail_*` - Email operations
- `mcp__google-workspace__calendar_*` - Calendar operations
- `mcp__google-workspace__drive_*` - File operations

**World Weaver MCP Tools**:
- `mcp__ww-memory__store_episode` - Store episodes
- `mcp__ww-memory__recall_episodes` - Retrieve memories
- `mcp__ww-memory__semantic_recall` - Entity search
- `mcp__ww-memory__recall_skill` - Skill lookup

## Data Flow

```
Voice Input
    │
    ├─ VoiceIntentParser
    │   ├─ Regex patterns (fast)
    │   └─ LLM fallback (complex)
    │
    ├─ VoiceMemoryBridge.get_relevant_context()
    │   ├─ Recent episodes (5)
    │   ├─ Related entities (10)
    │   └─ Applicable skills (3)
    │
    ├─ VoiceActionRouter.route()
    │   ├─ Permission check
    │   ├─ Handler execution
    │   └─ Episode storage
    │
    └─ VoiceResponse
        └─ Spoken text + action result
```

## Privacy & Security

- **PrivacyFilter**: PII redaction before storage
- **MemoryGate**: Automatic store/skip decisions
- **Permission Levels**: BLOCKED, CONFIRM, VERIFY, ALLOW
- **Confirmation Phrases**: "yes", "confirm", "do it" / "no", "cancel"
- **Voice Commands**: "off the record", "forget that"

## Installation

```bash
pip install -e ".[integrations]"
```

**Dependencies**:
- `httpx` - Async HTTP client
- `pydantic` - Data validation

## Public API

```python
# Google Workspace
GoogleWorkspaceSync, PersonalDataStore

# Kymera Voice
VoiceMemoryBridge, VoiceContext, MemoryContext
VoiceIntentParser, ParsedIntent
VoiceActionRouter, VoiceActionResult
VoiceActionExecutor, JarvisIntegration
JarvisMemoryHook
ContextInjector
ConversationCapture, ConversationSummarizer
PersonalDataManager, PersonalDataConfig

# Advanced
NotificationManager, PreferenceLearner
MultiModalContext, VoiceTriggerManager
```

## Integration Architecture

```
┌─ Kymera Voice (Jarvis) ──────────────────────────────────┐
│                                                           │
├─ JarvisMemoryHook (integration point)                    │
│  └─ VoiceActionExecutor (orchestrator)                   │
│     ├─ VoiceIntentParser (intent → action)               │
│     ├─ VoiceActionRouter (handler dispatch)              │
│     ├─ VoiceMemoryBridge (storage decisions)             │
│     └─ ContextInjector (prompt enhancement)              │
│                                                           │
├─ GoogleWorkspaceSync (personal data)                     │
│  └─ PersonalDataManager (search/resolution)              │
│                                                           │
└─ World Weaver Memory APIs                                │
   ├─ Episode storage/recall                               │
   ├─ Semantic search                                      │
   └─ Skill precondition checking                          │
```
