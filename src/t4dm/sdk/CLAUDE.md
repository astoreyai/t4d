# SDK
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/sdk/`

## What
Python SDK providing sync and async REST API clients for the T4DM memory system, plus a Claude Agent SDK integration for building learning agents.

## How
- **Client** (`client.py`): `T4DMClient` (sync) and `AsyncT4DMClient` (async) using httpx. Methods for storing/recalling episodes, entities, and skills. Custom exceptions: `ConnectionError`, `NotFoundError`, `RateLimitError`.
- **Models** (`models.py`): Pydantic models for API payloads -- `Episode`, `Entity`, `Skill`, `RecallResult`, `ActivationResult`, `HealthStatus`, `MemoryStats`.
- **Agent** (`agent.py`): `WWAgent` with `AgentPhase` lifecycle and `AgentConfig`. Wraps memory operations for agent reasoning loops.
- **Agent Client** (`agent_client.py`): `AgentMemoryClient` with `RetrievalContext`, `ScoredMemory`, and `CreditAssignmentResult`. Factory via `create_agent_memory_client()`.

## Why
Separates API consumption from server implementation. The agent client adds credit assignment and scored retrieval for reinforcement-learning-style agents that learn from memory feedback.

## Key Files
| File | Purpose |
|------|---------|
| `client.py` | Sync/async HTTP clients for REST API |
| `models.py` | Pydantic request/response models |
| `agent.py` | Agent lifecycle wrapper |
| `agent_client.py` | Memory client with credit assignment |

## Data Flow
```
Agent Code --> WWAgent --> AgentMemoryClient --> HTTP --> FastAPI --> Core Memory
                                             <-- RecallResult/ScoredMemory
```

## Integration Points
- **API**: Consumes REST endpoints defined in `t4dm.api`
- **Core**: Models mirror `t4dm.core.types` for serialization
- **Kymera**: Kymera agents use this SDK for memory access
