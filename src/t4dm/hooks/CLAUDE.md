# Hooks
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/hooks/`

## What
Extensible lifecycle hook system for all WW modules. Provides pre/post/on/error phase execution with priority ordering, error isolation, and async support.

## How
- **Hook** (ABC): Base class with `execute(context)`, priority, enable/disable, stats tracking.
- **HookRegistry**: Manages hooks per phase (PRE/POST/ON/ERROR). PRE hooks run sequentially (can modify input), POST hooks can run in parallel. Supports fail-fast or error-isolated modes.
- **HookContext**: Carries operation metadata, input/output data, session/user IDs, timing, and error state through the hook chain.
- **HookPriority**: CRITICAL(0) > HIGH(100) > NORMAL(500) > LOW(1000).
- **@with_hooks**: Decorator to wrap any async function with PRE/POST/ERROR hook execution.
- **Global registries**: Named registries per module (episodic, semantic, consolidation, storage_neo4j, etc.).

## Why
Decouples cross-cutting concerns (observability, caching, auditing, learning) from core memory operations. Any module can register hooks without modifying core logic.

## Key Files
| File | Purpose |
|------|---------|
| `base.py` | `Hook`, `HookRegistry`, `HookContext`, `HookPhase`, `HookPriority`, `@with_hooks` decorator |
| `registry.py` | Global singleton registries, standard registry names, `get_global_registry()` |
| `core.py` | `InitHook`, `ShutdownHook`, `HealthCheckHook`, `ConfigChangeHook` |
| `memory.py` | `CreateHook`, `RecallHook`, `UpdateHook`, `AccessHook`, `DecayHook` |
| `storage.py` | `ConnectionHook`, `QueryHook`, `ErrorHook`, `RetryHook` |
| `consolidation.py` | `ConsolidationHook`, `ClusterFormHook`, `DuplicateFoundHook`, `EntityExtractedHook` |
| `session_lifecycle.py` | `SessionStartHook`, `SessionEndHook`, `TaskOutcomeHook`, `IdleConsolidationHook` |

## Data Flow
```
Operation call -> PRE hooks (sequential, ordered by priority)
    -> Core function execution
    -> POST hooks (parallel or sequential)
    -> On error: ERROR hooks
```

## Integration Points
- **Every module**: All memory, storage, consolidation, and MCP operations use hooks
- **learning/**: Learning hooks emit retrieval/outcome events through this system
- **observability/**: Telemetry hooks registered via this system
