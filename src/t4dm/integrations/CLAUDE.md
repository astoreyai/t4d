# Integrations
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/integrations/`

## What
External service integrations that sync real-world data into WW memory. Contains Google Workspace sync and the Kymera voice assistant integration.

## How
### Google Workspace (`google_workspace.py`)
- **GoogleWorkspaceSync**: Uses MCP tools to fetch Gmail, Calendar, and Drive data. Converts raw API responses into typed entities (`Email`, `CalendarEvent`, `Contact`). Applies privacy filtering, generates embeddings, and builds aggregated `PersonalContext`.
- **PersonalDataStore**: Persists personal entities to dual-store (Neo4j nodes + Qdrant vectors) with semantic search across contacts, events, and emails.

### Kymera Voice (`kymera/`)
- **JarvisMemoryHook**: Primary integration point for Kymera voice assistant. Enhances voice context with relevant memories before Claude calls and stores interactions after.
- **VoiceMemoryBridge**: Connects voice sessions to WW memory retrieval/storage.
- **VoiceIntentParser**: Parses natural language intents with time expressions.
- **VoiceActionRouter/Executor**: Routes parsed intents to executable actions.
- **MemoryConsolidator**: Captures conversations, summarizes, and consolidates into long-term memory.
- **NotificationManager, PreferenceLearner**: Advanced features for proactive context and user preference learning.

## Why
Makes WW a practical personal AI by connecting to real data sources (email, calendar) and voice interfaces (Kymera). Personal context enables temporally-aware, relationship-aware memory retrieval.

## Key Files
| File | Purpose |
|------|---------|
| `google_workspace.py` | Gmail/Calendar/Drive sync, privacy filtering, embedding, `PersonalDataStore` |
| `kymera/jarvis_hook.py` | Main Kymera integration hook, context enhancement |
| `kymera/bridge.py` | Voice-to-memory bridge |
| `kymera/intent_parser.py` | NL intent and time parsing |
| `kymera/voice_actions.py` | Action execution, `JarvisIntegration` |
| `kymera/memory_continuity.py` | Conversation capture, summarization, consolidation |
| `kymera/personal_data.py` | Contact cache and personal data management |
| `kymera/context_injector.py` | Injects relevant memory context into conversations |

## Data Flow
```
Google Workspace APIs (via MCP) -> GoogleWorkspaceSync -> typed entities
    -> PrivacyFilter -> embeddings -> PersonalDataStore (Neo4j + Qdrant)

Voice input -> JarvisMemoryHook.enhance_context() -> WW recall -> enriched prompt
Claude response -> JarvisMemoryHook.on_response() -> episodic store -> consolidation
```

## Integration Points
- **core/personal_entities.py**: Entity type definitions (Email, Contact, CalendarEvent)
- **core/privacy_filter.py**: Content filtering before storage
- **memory/**: Episodic storage for conversations, semantic for entities
- **bridges/**: Neo4j and Qdrant dual-store persistence
